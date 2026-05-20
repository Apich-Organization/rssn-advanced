# RSSN-Advanced 开发计划（基于审查反馈）

本计划将 `review/` 目录下 10 份审查报告归并为可执行的工程任务。任务按 **依赖关系** 与 **风险等级** 分阶段执行：先建立横向基础设施（错误宏、零拷贝容器、纤程运行时），再切入垂直模块改造。

> **图例**：🔴 Critical（违反明确指令）  🟠 High（架构风险）  🟡 Medium（性能/可维护性）  🟢 Low（润色）

---

## 阶段 0：横向基础设施（Foundations）

这些是所有上层模块都会依赖的"地基"，必须最先完成。

### T0.1 🔴 `bincode_error!` 冷路径错误宏 — `src/error/mod.rs`
- **来源**：`ast_review §4`、`dag/ffi/heuristic/jit/parallel/parser/simd/storage_review §4`（全员违规）。
- **任务**：
  1. 新建 `src/error/mod.rs`，实现审查文档中给出的 `bincode_error!` macro_rules（带 `#[cold]` `#[inline(never)]` `#[track_caller]` 的 `cold_*` 构造函数）。
  2. 在 `lib.rs` 暴露 `pub use error::bincode_error;`。
  3. 定义模块级错误枚举：`AstError`、`DagError`、`JitError`、`ParallelError`、`ParserError`、`StorageError`、`FfiError`。
  4. 编写单元测试，验证宏展开后 `cold_*` 函数存在且确实带 `#[cold]`。
- **验收**：所有 `.expect()` / `.unwrap()` / `.assert_eq!` 在后续阶段被替换为 `cold_*` 调用。
- **预估**：0.5 天。

### T0.2 🔴 零拷贝容器抽象 — `src/zerocopy/mod.rs`
- **来源**：`ast_review §3`、`dag_review §3`、`storage_review §1,§3`。
- **背景**：当前 `Vec<DagNode>`、`Vec<RelPtr>` 无法 `BorrowDecode`。
- **任务**：
  1. 设计 `BorrowedSlice<'a, T: Pod>`，实现 `bincode_next::de::BorrowDecode<'a>`，内部为 `&'a [T]`（要求 `T: bytemuck::Pod` 或 `#[repr(C)]`）。
  2. 设计 `BorrowedArena<'a, Node>`：头部 `len: u32` + 紧跟节点数组。
  3. 实现 `mmap` 支持的 `MappedArena<Node>`（Linux/Windows 双路径，已有 `windows-sys` 依赖）。
  4. 提供 `OwnedArena ↔ BorrowedArena` 互转 helper。
- **验收**：单元测试：序列化一个 1M 节点的 owned arena → mmap 读取 → 反序列化耗时与拷贝量分别测量。
- **预估**：2 天。

### T0.3 🔴 `dtact` 纤程执行器封装 — `src/runtime/mod.rs`
- **来源**：`ffi_review §1`、`parallel_review §2`。
- **任务**：
  1. 在 `src/runtime/` 下封装 `dtact` 的任务派发原语 `spawn_task<F>(f: F)`、`join_all`。
  2. 提供共享全局 executor 句柄（首次调用懒初始化，避免 `OnceLock` 锁竞争——使用 `std::sync::OnceLock`）。
  3. 提供 `parallel_for_each<I, F>(iter, f)` 高级 API，供 `parallel`/`ffi` 模块复用。
- **验收**：替换至少一处 `std::thread::spawn` 后基准测试任务延迟下降 ≥3×。
- **预估**：1 天。

### T0.4 🟠 工作栈（Work-list）通用工具 — `src/util/worklist.rs`
- **来源**：`ast_review §2`、`dag_review`（隐含）、`heuristic_review §2`、`jit_review §2`、`parallel_review §2`、`parser_review §2`。
- **任务**：实现迭代后序遍历的通用 helper：
  ```rust
  pub fn post_order<N, ID, FChildren, FVisit>(root: ID, children: FChildren, mut visit: FVisit)
  where ID: Copy + Eq + Hash,
        FChildren: Fn(ID) -> &[ID],
        FVisit: FnMut(ID);
  ```
- **预估**：0.5 天。

---

## 阶段 1：核心数据结构改造（DAG/AST 零拷贝）

依赖阶段 0 全部完成。

### T1.1 🔴 `DagArena` 零拷贝重构
- **来源**：`dag_review §3`、`storage_review §1`。
- **任务**：
  1. 将 `DagNode` 改为 `#[repr(C)]` + 实现 `bytemuck::Pod`。
  2. `DagArena` 内部改为 `Cow<'a, [DagNode]>` 或两套类型 `OwnedDagArena` / `BorrowedDagArena<'a>`。
  3. 用 `BorrowDecode` 实现零拷贝反序列化。
  4. 保留 `DagBuilder` API 兼容。
- **预估**：2 天。

### T1.2 🟠 紧凑节点表示（KISS / 缓存友好）
- **来源**：`dag_review §2`。
- **现状（2026-05-17 更新）**：**Wire 形态已落地**，in-memory 形态延后。
  - ✅ `PackedDagNode`（32 字节，`#[repr(C)] + Pod`）已在 `src/dag/packed.rs` 实现并验证大小（`const _: () = assert!(size_of::<PackedDagNode>() == 32);`）。
  - ✅ `coefficient` 与常量值合并到同一字段（由 `kind_tag` 区分），消除 `Option<f64>`。
  - ✅ 多于 2 个子节点溢出到共享 `children_pool: Vec<u32>`，避免每个节点单独分配 `Vec`。
  - ⏸ **延后**：将运行时 `DagNode` 直接替换为 `PackedDagNode`。当前消费者（jit, parallel, heuristic, parser tests）共 58 处直接访问 `node.kind` / `.value` / `.children` / `.meta`，一次性切换风险大；待 Phase 2/4 引入 packed accessor 后再批量迁移。
- **后续任务**：
  1. 添加 `DagArena::with_packed_storage` 配置（或独立 `PackedDagArena`），让追求极致 cache 性能的工作流可选 32 字节布局。
  2. 把 jit/parallel/heuristic 三大循环改为通过 `PackedDagNode` 的 accessor 方法（`kind()` / `value()` / `meta()` / `Children`）访问，逐步迁移消费者。
  3. 在迁移完成后，再删除 `dag::node::DagNode` 富类型。
- **预估**：剩余约 1 天（迁移消费者）。

### T1.3 🟠 `SymbolRegistry` O(1) 化
- **来源**：`dag_review §2`。
- **任务**：用 `HashMap<Box<str>, SymbolId>`（或 `rapidhash` 已有依赖）替换线性扫描。保留稳定 `SymbolId` 顺序。
- **预估**：0.25 天。

### T1.4 🔴 `AstProjection` 真正栈本地化
- **来源**：`ast_review §1,§2`。
- **任务**：
  1. 定义 `AstProjection<'arena, const N: usize>`，节点存于内嵌 `ArrayVec<AstNode, N>`，超出走 scratchpad 分配器。
  2. `AstChildList::Many` 改为 `&'arena [RelPtr<AstNode>]` 借用 scratchpad。
  3. `convert_dag_node` / `convert_ast_node` 改为基于 T0.4 的迭代实现。
- **预估**：2 天。

---

## 阶段 2：JIT 与硬件级优化

依赖阶段 0/1。

### T2.1 🔴 JIT 迭代代码生成
- **来源**：`jit_review §2`。
- **任务**：改写 `src/jit/compiler.rs::compile_node` 为基于 T0.4 worklist 的迭代版本；DAG 后序遍历 → SSA value 出栈合成。
- **预估**：1.5 天。

### T2.2 🔴 显式 `prefetch` 指令插入
- **来源**：`jit_review §1`（`emit_prefetch_hint` 已存在但从未调用）。
- **任务**：
  1. 在 `compile_node` 访问子节点元数据前插入 `emit_prefetch_hint`（基于 Cranelift `prefetch` intrinsic，若无则降级为 `inline_asm!("prefetcht0")`）。
  2. 添加 prefetch 距离参数（默认 8 节点）。
- **预估**：0.5 天。

### T2.3 🔴 JIT `Acquire/Release` 原子序
- **来源**：`jit_review §1`。
- **任务**：JIT 缓存读写、热点表交互处嵌入 `AtomicOrdering::{Acquire, Release}`；CI 添加 `grep -nE 'SeqCst'` 拒绝合并的检查脚本。
- **预估**：0.5 天。

### T2.4 🔴 Naked / Inline ASM 预设套件 — `src/jit/asm_presets/`
- **来源**：`jit_review §1`、`simd_review §1`。
- **任务**：实现以下预设（每个文件一个 kernel）：
  - `add_f64x4_avx2.rs` `mul_f64x4_avx2.rs` `fma_f64x4_avx2.rs`
  - `hash_u64x2_aesni.rs`
  - `coef_merge_f64x4.rs`（系数合并）
  - `cmp_eq_f64x4.rs`
  每个文件提供 `#[cfg(target_arch = "x86_64")]` 实现 + scalar fallback。使用 `core::arch::asm!`，**禁止仅依赖 auto-vectorization**。
- **验收**：`cargo asm` 验证生成的指令包含 `vfmadd231pd` / `vpxor` 等目标指令。
- **预估**：3 天。

### T2.5 🟠 JIT 系数合并 & 恒等化简
- **来源**：`jit_review §1,§2`。
- **任务**：
  1. 在 IR 生成前进行一遍 peephole：`(c1 * x) * (c2 * y) → (c1*c2) * (x*y)`；`x + 0 → x`；`x * 1 → x`；`x * 0 → 0`。
  2. 调用 `src/jit/primitives.rs` 里既有但未使用的 `simplify_add` / `simplify_mul`。
- **预估**：1 天。

### T2.6 🟢 自定义函数 JIT 支持
- **来源**：`jit_review §5`。
- **任务**：`SymbolKind::Function` 通过 `JitCustomRegistry` 注册回调（Cranelift `import_function`）。
- **预估**：1 天。

---

## 阶段 3：SIMD 完整套件

### T3.1 🔴 显式 `std::arch` 内核
- **来源**：`simd_review §1,§2`。
- **任务**：重写 `simd/arithmetic.rs`、`simd/hash.rs`：
  1. 使用 `core::arch::x86_64::*` 显式 AVX2/FMA 内联。
  2. 对应 fallback 走 T2.4 的 scalar 版本，**不再**依赖编译器循环向量化。
  3. 补全 `batch_pow`、`batch_cmp_eq`、`batch_coef_merge`。
- **预估**：2 天。

---

## 阶段 4：并行 & 启发式（核心架构修复）

### T4.1 🔴 `parallel` 共享 Arena（去除克隆）
- **来源**：`parallel_review §2`、`summary_review §2`。
- **任务**：
  1. `parallel_evaluate` 签名改为 `fn(arena: &DagArena, ...)`，使用 `Arc<DagArena>` 或纯 `&` 借用（计算路径 arena 只读）。
  2. 删除所有 `arena.clone()`。
  3. 用 T0.3 的 `parallel_for_each` 替换 `thread::spawn`。
  4. `evaluate_node` 改迭代版本。
- **验收**：1M 节点 / 16 chunks 的内存占用从 ~1.4GB 降到 ~90MB。
- **预估**：2 天。

### T4.2 🔴 `HeuristicEngine` 强制走 `DagBuilder`
- **来源**：`heuristic_review §1`、`summary_review §3.4`。
- **任务**：
  1. `HeuristicEngine` 持有 `&mut DagBuilder`（不是 `&mut DagArena`）。
  2. 所有 `arena.alloc(new_node)` 改为 `builder.get_or_insert(new_node)`。
  3. `explore_and_rewrite` / `approximate_simplify_rec` 改迭代。
  4. 实现真正的 pattern matching：`x+0`、`x*1`、`x*0`、`x-x`、`x/x` 等基础规则放进 `src/heuristic/patterns.rs`，以 trait 注册。
  5. `approximate_simplify` 改为基于"低系数项剪枝"，而不是无意义地折叠到 1.0。
- **预估**：3 天。

### T4.3 🟡 阶段性化简实现
- **来源**：`parallel_review §1`。
- **任务**：实现 `SimplifyConfig` 的"每 N 轮触发一次本地化简"逻辑（已有配置无实现）。
- **预估**：0.5 天。

---

## 阶段 5：存储 & 热点表

### T5.1 🔴 `DiskCache` 真正流式 / mmap
- **来源**：`storage_review §1`、`summary_review §2`。
- **任务**：
  1. `load` 改用 T0.2 提供的 `MappedArena` mmap。
  2. 暴露 `load_borrowed<'a>(&'a self) -> BorrowedDagArena<'a>`。
  3. 删除 `read_to_end`。
- **预估**：1 天。

### T5.2 🟠 `DynamicHotspotTable` 去锁
- **来源**：`storage_review §2`。
- **任务**：
  1. 改造为线程局部计数器（`thread_local!`）+ 周期性汇总到共享 `dashmap` 或自实现的分片表（16 shard，每 shard 内部 `RwLock`）。
  2. 用 `#[repr(align(128))]` 隔离每个 shard 避免 false sharing。
- **预估**：1.5 天。

### T5.3 🔴 修复 Eviction 悬挂引用 Bug
- **来源**：`storage_review §2`。
- **任务**：实现基于"热点节点传递闭包"的标记-清除：先收集 hot roots，DFS 标记它们的所有子孙为 protected，再驱逐未标记冷节点。
- **验收**：Proptest 生成随机 DAG → 模拟随机访问模式 → 驱逐后断言无悬空 `DagNodeId`。
- **预估**：1.5 天。

---

## 阶段 6：FFI & Parser 润色

### T6.1 🔴 FFI 换 `dtact`
- **来源**：`ffi_review §1`。
- **任务**：`src/ffi/async_bridge.rs` 用 T0.3 的 runtime 替换 `std::thread::spawn`。
- **预估**：0.5 天。

### T6.2 🟡 FFI 错误返回值统一
- **来源**：`ffi_review §1`。
- **任务**：所有 `rssn_dag_*` 函数改为 `RssnStatus` + out-param（`u32* out_id`），消除 `u32::MAX` 哨兵约定。同步更新 cbindgen 生成的头文件与 `examples/`。
- **预估**：1 天。

### T6.3 🟡 FFI 字符串零分配
- **来源**：`ffi_review §2`。
- **任务**：`rssn_dag_variable` 用 `CStr::to_bytes` 在 `SymbolRegistry` 里查找，避免 `to_string_lossy()`。需要 `SymbolRegistry` 支持 `lookup_by_bytes(&[u8])`。
- **预估**：0.5 天。

### T6.4 🟡 Parser 迭代化 atom
- **来源**：`parser_review §2`。
- **任务**：将 `parse_atom` 中的括号处理转为显式深度栈，限制 `MAX_PAREN_DEPTH = 1024`。
- **预估**：0.5 天。

### T6.5 🟢 Parser 错误位置改 line:col
- **来源**：`parser_review §3`。
- **任务**：保留原始 buffer 引用 + 计算 line/col。`ParseError::span` 改为 `Span { line: u32, col: u32, len: u32 }`。
- **预估**：0.5 天。

---

## 阶段 7：错误宏全量替换（收尾）

### T7.1 🔴 全模块替换 `unwrap` / `expect` / `assert_eq` → `cold_*`
- **来源**：所有审查报告 §4。
- **任务**：grep 全仓库 `\.unwrap\(\)|\.expect\(|assert_eq!` 并按模块迁移。CI 添加 `clippy::unwrap_used` 拒绝合并。
- **预估**：2 天。

---

## 阶段 8：扩展性（推迟到 MVP 之后）

以下不影响审查所列的 Critical / High 问题，可作为 v0.1 之后的目标：

- 🟢 `SymbolKind` 改 trait-based 注册（`ast_review §5`、`dag_review §5`、`parallel_review §5`）。
- 🟢 Parser 自定义运算符注册（`parser_review §4`）。
- 🟢 FFI 引入 "request/response" 风格批量接口（`ffi_review §5`）。

---

## 阶段汇总与排程

| 阶段 | 关键产出 | 估时 | 阻塞下游 |
| :--- | :--- | :--- | :--- |
| 0 基础设施 | error 宏 / 零拷贝容器 / dtact runtime / worklist | 4 天 | 全部 |
| 1 数据结构 | DAG/AST 零拷贝 + 紧凑布局 | 5.75 天 | 2,3,4,5 |
| 2 JIT | 迭代 codegen + ASM 预设 + prefetch + 原子序 | 7.5 天 | 3 |
| 3 SIMD | 显式 `std::arch` + 完整套件 | 2 天 | 4 |
| 4 并行/启发式 | 共享 arena + dedup + pattern matching | 5.5 天 | 5 |
| 5 存储 | mmap 流式 + 分片热点表 + 安全驱逐 | 4 天 | 6 |
| 6 FFI/Parser | dtact / 错误统一 / 迭代 atom | 3 天 | 7 |
| 7 错误收尾 | 全仓库 cold_* 替换 | 2 天 | — |
| **MVP 合计** | | **~34 天** | |
| 8 扩展性 | trait 化运算符 / 自定义算子 | TBD | — |

---

## 验收清单（与 `summary_review.md` 顶层 6 条对齐）

合并主分支前必须全部 ✅：

1. ✅ `DagArena` / `AstProjection` 支持 `BorrowDecode`（T0.2 + T1.1 + T1.4）。
2. ✅ FFI 与 parallel 不再出现 `std::thread::spawn`（T0.3 + T4.1 + T6.1）。
3. ✅ SIMD/JIT 至少 6 个 `inline_asm!` 预设可用（T2.4 + T3.1）。
4. ✅ 启发式模块 100% 走 `DagBuilder::get_or_insert`（T4.2）。
5. ✅ 全部递归遍历路径替换为 worklist（T1.4 + T2.1 + T4.1 + T4.2 + T6.4）。
6. ✅ `bincode_error!` 宏存在且全仓库无裸 `unwrap`（T0.1 + T7.1）。
