//! WGSL compute shader compiler and WebGPU execution runtime.
//!
//! Provides the AST-to-WGSL translation pass and the wgpu buffer dispatch
//! logic for executing JIT compiled expressions on the GPU.

use crate::ast::projection::AstProjection;
use crate::dag::symbol::{OpKind, SymbolKind, SymbolRegistry};
use bytemuck;

/// JIT compiles the given AST projection into a fully-functional WebGPU WGSL compute shader.
///
/// Automatically registers all variable bindings, maps constant literals, and resolves
/// transcendental/intrinsic functions to their native GPU hardware shader equivalents.
pub fn compile_to_wgsl(
    ast: &AstProjection,
    var_registry: &SymbolRegistry,
    fn_registry: &SymbolRegistry,
    var_order: &[&str],
) -> Result<String, String> {
    if ast.is_empty() {
        return Err("Cannot compile empty AST projection to WGSL".to_string());
    }

    // Format the root node recursively.
    let expr_str = format_node(ast, 0, var_registry, fn_registry, var_order)?;

    // Construct the storage bindings and shader entry point template.
    let mut bindings = String::new();
    for (i, _) in var_order.iter().enumerate() {
        bindings.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read> var_{i}: array<f32>;\n"
        ));
    }
    // The output buffer is bound directly after the variables.
    let out_binding = var_order.len();
    bindings.push_str(&format!(
        "@group(0) @binding({out_binding}) var<storage, read_write> out_col: array<f32>;\n"
    ));

    let wgsl_code = format!(
        "{bindings}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= arrayLength(&out_col)) {{
        return;
    }}

    let val = {expr_str};
    out_col[idx] = val;
}}
"
    );

    Ok(wgsl_code)
}

/// Recursive helper to translate an AST node into a WGSL expression string.
fn format_node(
    projection: &AstProjection,
    node_idx: usize,
    var_registry: &SymbolRegistry,
    fn_registry: &SymbolRegistry,
    var_order: &[&str],
) -> Result<String, String> {
    let node = projection
        .nodes
        .get(node_idx)
        .ok_or_else(|| "Invalid node index in AST projection".to_string())?;

    match node.kind {
        SymbolKind::Constant(v) => {
            // WebGPU requires precise float representation for constants.
            if v.is_infinite() || v.is_nan() {
                return Err("GPU JIT does not support infinity or NaN constants".to_string());
            }
            // Ensure valid floating-point literal representation in WGSL (force a decimal place).
            if v.fract() == 0.0 {
                Ok(format!("{v:.1}f"))
            } else {
                Ok(format!("{v}f"))
            }
        }
        SymbolKind::Variable(sym_id) => {
            let name = var_registry
                .name(sym_id)
                .ok_or_else(|| "Unknown variable symbol ID".to_string())?;
            let var_idx = var_order
                .iter()
                .position(|&s| s == name)
                .ok_or_else(|| format!("Variable '{name}' not found in var_order"))?;
            Ok(format!("var_{var_idx}[idx]"))
        }
        SymbolKind::Operator(op) => {
            let children = node.children.as_slice_with_pool(&projection.children_pool);
            match op {
                OpKind::Neg => {
                    if children.len() != 1 {
                        return Err("Neg operator must have exactly 1 child".to_string());
                    }
                    let child_idx = children[0]
                        .resolve(node_idx)
                        .ok_or_else(|| "Unresolved relative child pointer".to_string())?;
                    let inner =
                        format_node(projection, child_idx, var_registry, fn_registry, var_order)?;
                    Ok(format!("(-{inner})"))
                }
                OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Mod => {
                    if children.len() != 2 {
                        return Err(format!("{op:?} operator must have exactly 2 children"));
                    }
                    let lhs_idx = children[0]
                        .resolve(node_idx)
                        .ok_or_else(|| "Unresolved left child pointer".to_string())?;
                    let rhs_idx = children[1]
                        .resolve(node_idx)
                        .ok_or_else(|| "Unresolved right child pointer".to_string())?;
                    let lhs =
                        format_node(projection, lhs_idx, var_registry, fn_registry, var_order)?;
                    let rhs =
                        format_node(projection, rhs_idx, var_registry, fn_registry, var_order)?;
                    let op_str = match op {
                        OpKind::Add => "+",
                        OpKind::Sub => "-",
                        OpKind::Mul => "*",
                        OpKind::Div => "/",
                        OpKind::Mod => "%",
                        _ => unreachable!(),
                    };
                    Ok(format!("({lhs} {op_str} {rhs})"))
                }
                OpKind::Pow => {
                    if children.len() != 2 {
                        return Err("Pow operator must have exactly 2 children".to_string());
                    }
                    let lhs_idx = children[0]
                        .resolve(node_idx)
                        .ok_or_else(|| "Unresolved left child pointer".to_string())?;
                    let rhs_idx = children[1]
                        .resolve(node_idx)
                        .ok_or_else(|| "Unresolved right child pointer".to_string())?;
                    let lhs =
                        format_node(projection, lhs_idx, var_registry, fn_registry, var_order)?;
                    let rhs =
                        format_node(projection, rhs_idx, var_registry, fn_registry, var_order)?;
                    Ok(format!("pow({lhs}, {rhs})"))
                }
            }
        }
        SymbolKind::Function(fn_id) => {
            let children = node.children.as_slice_with_pool(&projection.children_pool);
            let fn_name = fn_registry
                .name(crate::dag::symbol::SymbolId(fn_id.0))
                .ok_or_else(|| "Unknown function symbol ID".to_string())?;

            // Map standard transcendentals to their WGSL built-in shader functions.
            let wgsl_fn = match fn_name {
                "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh"
                | "exp" | "log" | "log2" | "sqrt" | "abs" | "floor" | "ceil" | "round" | "sign" => {
                    fn_name
                }
                _ => return Err(format!("Unsupported function on GPU: '{fn_name}'")),
            };

            let mut args = Vec::new();
            for ptr in children {
                let child_idx = ptr
                    .resolve(node_idx)
                    .ok_or_else(|| "Unresolved relative child pointer".to_string())?;
                args.push(format_node(
                    projection,
                    child_idx,
                    var_registry,
                    fn_registry,
                    var_order,
                )?);
            }
            Ok(format!("{}({})", wgsl_fn, args.join(", ")))
        }
        SymbolKind::ControlFlow(_) => {
            Err("Control flow is not supported on the GPU backend".to_string())
        }
    }
}

/// Holds compiled GPU resources together to prevent recreation and layout mismatch overhead.
pub struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// A high-performance WebGPU compute execution context with dynamic buffer pooling and pipeline caching.
pub struct GpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_cache: std::collections::HashMap<u64, CachedPipeline, rapidhash::fast::GlobalState>,
    input_buffers: Vec<wgpu::Buffer>,
    output_buffer: Option<wgpu::Buffer>,
    staging_buffer: Option<wgpu::Buffer>,
    /// Reusable CPU buffer to avoid memory allocations during f64 -> f32 conversions on every batch call.
    conversion_scratchpad: Vec<f32>,
    current_capacity: usize,
}

/// Helper to compute rapidhash of WGSL shader source.
fn hash_shader_src(src: &str) -> u64 {
    use std::hash::Hasher;
    let mut hasher = rapidhash::fast::RapidHasher::default();
    hasher.write(src.as_bytes());
    hasher.finish()
}

impl GpuExecutor {
    /// Discovers and initialises the default GPU adapter and device.
    ///
    /// Returns `None` if no compatible GPU hardware backend is available.
    #[must_use]
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = match pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        ) {
            Ok(adapter) => adapter,
            Err(_) => return None,
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RSSN JIT GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        Some(Self {
            device,
            queue,
            pipeline_cache: std::collections::HashMap::default(),
            input_buffers: Vec::new(),
            output_buffer: None,
            staging_buffer: None,
            conversion_scratchpad: Vec::new(),
            current_capacity: 0,
        })
    }

    /// Dispatches a high-throughput parallel compute job to the GPU using the compiled WGSL shader source.
    ///
    /// Performs zero-allocation float array conversions at the CPU-GPU memory boundary
    /// using a pre-allocated conversion scratchpad, maintaining optimal cache locality.
    pub fn execute_batch(
        &mut self,
        shader_src: &str,
        n_rows: usize,
        vars_cols: &[&[f64]],
        out: &mut [f64],
    ) -> Result<(), String> {
        if n_rows == 0 {
            return Ok(());
        }

        // 1. Get or Compile WGSL Shader Module & Compute Pipeline (utilising Fast LRU hash cache)
        let hash = hash_shader_src(shader_src);
        if !self.pipeline_cache.contains_key(&hash) {
            let shader_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("RSSN JIT Compute Shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

            let compute_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("RSSN JIT GPU Compute Pipeline"),
                        layout: None,
                        module: &shader_module,
                        entry_point: Some("main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

            // Cache bind group layout along with pipeline to optimize driver-level validation passes
            let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

            self.pipeline_cache.insert(
                hash,
                CachedPipeline {
                    pipeline: compute_pipeline,
                    bind_group_layout,
                },
            );
        }
        let cached = self
            .pipeline_cache
            .get(&hash)
            .ok_or("Failed to fetch compute pipeline")?;

        // 2. Stateful Buffer Pooling: Re-allocate ONLY when row capacity changes
        let needs_realloc = self.output_buffer.is_none() || self.current_capacity != n_rows;
        if needs_realloc {
            self.current_capacity = n_rows;
            let output_size_bytes = (n_rows * std::mem::size_of::<f32>()) as u64;

            self.output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Output Storage Buffer"),
                size: output_size_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));

            self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Staging Buffer"),
                size: output_size_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));

            // Clear input buffers to trigger a resize/reallocation
            self.input_buffers.clear();
        }

        // Dynamically scale/allocate input buffers if variable column size changes (safely handles up and down scaling)
        if self.input_buffers.len() != vars_cols.len() {
            self.input_buffers.clear();
            let input_size_bytes = (n_rows * std::mem::size_of::<f32>()) as u64;
            for i in 0..vars_cols.len() {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("GPU Input Storage Buffer {i}")),
                    size: input_size_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.input_buffers.push(buffer);
            }
        }

        // 3. Write input columns into pooled GPU memory buffers (zero-allocation)
        // Re-use scratchpad space to avoid per-column vec-allocations
        self.conversion_scratchpad.resize(n_rows, 0.0f32);
        for (i, col) in vars_cols.iter().enumerate() {
            // Unroll slightly or optimize via element copying
            for (dst, &src) in self.conversion_scratchpad.iter_mut().zip(col.iter()) {
                *dst = src as f32;
            }
            self.queue.write_buffer(
                &self.input_buffers[i],
                0,
                bytemuck::cast_slice(&self.conversion_scratchpad),
            );
        }

        // 4. Map Bind Group Entries (zero-copy references from stateful buffers)
        let mut bind_group_entries = Vec::with_capacity(vars_cols.len() + 1);
        for (i, buffer) in self.input_buffers.iter().enumerate() {
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        let output_buffer = self
            .output_buffer
            .as_ref()
            .ok_or("Uninitialized output buffer")?;
        let staging_buffer = self
            .staging_buffer
            .as_ref()
            .ok_or("Uninitialized staging buffer")?;

        // Output binding is placed immediately after input bindings.
        let out_binding = vars_cols.len() as u32;
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: out_binding,
            resource: output_buffer.as_entire_binding(),
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RSSN JIT GPU Bind Group"),
            layout: &cached.bind_group_layout,
            entries: &bind_group_entries,
        });

        // 5. Command Encoding and Pass Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RSSN JIT GPU Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RSSN JIT Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&cached.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch in workgroups of size 256.
            let workgroup_count = n_rows.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        let output_size_bytes = (n_rows * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, output_size_bytes);
        self.queue.submit(Some(encoder.finish()));

        // 6. Map staging buffer with explicit bounds and fetch computed results.
        let buffer_slice = staging_buffer.slice(0..output_size_bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Poll device until mapping completes.
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        receiver
            .recv()
            .map_err(|e| format!("GPU channel disconnect error: {e}"))?
            .map_err(|e| format!("Staging buffer mapping failed: {e:?}"))?;

        {
            let data = buffer_slice.get_mapped_range();
            let f32_out: &[f32] = bytemuck::cast_slice(&data);

            // Convert results back to f64 matching our JIT execution signature.
            for (dest, &src) in out.iter_mut().zip(f32_out.iter()) {
                *dest = f64::from(src);
            }
        }

        staging_buffer.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::convert::dag_to_ast;
    use crate::dag::builder::DagBuilder;
    use crate::parser::expr::parse_expression;

    #[test]
    fn test_compile_simple_expression_to_wgsl() {
        let mut builder = DagBuilder::new();
        let root = parse_expression("x^2 + 2*x + 1.5", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);
        let var_order = vec!["x"];

        let shader =
            compile_to_wgsl(&ast, builder.registry(), builder.fn_registry(), &var_order).unwrap();

        assert!(shader.contains("@group(0) @binding(0) var<storage, read> var_0: array<f32>;"));
        assert!(
            shader.contains("@group(0) @binding(1) var<storage, read_write> out_col: array<f32>;")
        );
        assert!(shader.contains("let val = "));
    }

    #[test]
    fn test_execute_batch_gpu() {
        let mut executor = match GpuExecutor::new() {
            Some(e) => e,
            None => return, // Skip test if no GPU compatible hardware is found
        };

        let mut builder = DagBuilder::new();
        let root = parse_expression("x + y + 10.0", &mut builder).unwrap();
        let ast = dag_to_ast(builder.arena(), root);
        let var_order = vec!["x", "y"];

        let shader =
            compile_to_wgsl(&ast, builder.registry(), builder.fn_registry(), &var_order).unwrap();

        let n = 1000;
        let x_data = vec![1.0; n];
        let y_data = vec![2.0; n];
        let mut out = vec![0.0; n];

        // Execute batch multiple times to verify caching and buffer reuse works flawlessly
        for _ in 0..3 {
            executor
                .execute_batch(&shader, n, &[&x_data, &y_data], &mut out)
                .unwrap();
            for i in 0..n {
                assert_eq!(out[i], 13.0); // 1.0 + 2.0 + 10.0
            }
        }
    }
}
