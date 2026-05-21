# RSSN-Advanced Review: `src/ast`

## **1. Alignment with `plan.md`**

### **Relative Pointers**
- **Status**: **PASS (Partial)**
- **Observation**: `RelPtr` correctly implements the relative pointer strategy mentioned in the plan (using `i32` or `i64`).
- **Issue**: The plan suggests these should be used in "stack-local" trees. While `AstProjection` uses them, the underlying storage is still a heap-allocated `Vec<AstNode>`.

### **Local AST Projection**
- **Status**: **FAIL**
- **Observation**: The plan emphasizes a "stack-local projection tree". Current implementation of `AstProjection` relies on `Vec<AstNode>`, and `AstChildList::Many` uses `Vec<RelPtr<AstNode>>`. These force heap allocations, defeating the purpose of a "stack-local" cache-friendly tree for JIT functions.

---

## **2. Performance & Memory Issues**

### **Heap Spillage in "Stack-Local" Structures**
- **Issue**: `AstChildList::Many(Vec<RelPtr<AstNode>>)` introduces heap allocation for nodes with >4 children. In high-performance symbolic computation (e.g., large sums or products), this will trigger frequent allocations during what should be a "stack-local" projection.
- **Recommendation**: Use a fixed-size inline buffer with a fallback to a scratchpad allocator, or ensure the entire `AstProjection` buffer is pre-allocated on the stack or via a pool.

### **Conversion Recursion**
- **Issue**: `convert_dag_node` and `convert_ast_node` are recursive.
- **Risk**: For deep symbolic expressions (e.g., `x + (x + (x + ...))`), this will lead to a **Stack Overflow**.
- **Recommendation**: Implement iterative conversion using an explicit work-stack.

---

## **3. Zero-Copy & `bincode-next`**

### **Lack of Zero-Copy Decoding**
- **Issue**: The user explicitly reminded that the project "shall use the zero-copy feature of bincode-next".
- **Evidence**: `AstProjection` and `AstNode` implement `Decode`, but they do not leverage `BorrowDecode` or `Borrowed` types. Specifically, `AstChildList::Many(Vec<...>)` cannot be zero-copy decoded because `Vec` owns its data.
- **Recommendation**: Refactor `AstChildList` to use a slice `&'a [RelPtr<AstNode>]` or a specialized zero-copy collection that can borrow from the input buffer.

---

## **4. Error Handling**

### **Missing Cold-Path Error Macro**
- **Issue**: The project is instructed to use a specific `rssn_error!`-style macro for cold-path error handling.
```rust
#[doc(hidden)]
#[macro_export]
macro_rules! rssn_error {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident $( { $( $(#[$field_meta:meta])* $field:ident : $ftype:ty ),* $(,)? } )? $( ( $( $(#[$tuple_meta:meta])* $tname:ident : $ttype:ty ),* $(,)? ) )?
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant $( { $( $(#[$field_meta])* $field : $ftype ),* } )? $( ( $( $(#[$tuple_meta])* $ttype ),* ) )?,
            )*
        }

        pastey::paste! {
            $(
                $(#[$variant_meta])*
                #[doc(hidden)]
                #[cold]
                #[track_caller]
                #[inline(never)]
                pub const fn [<cold_ $name:snake _ $variant:snake>]<T>(
                    $($($field : $ftype),*)?
                    $($($tname : $ttype),*)?
                ) -> core::result::Result<T, $name> {
                    core::result::Result::Err($name::$variant $( { $($field),* } )? $( ( $( $tname ),* ) )?)
                }
            )*
        }
    };
}
```
- **Observation**: Current code uses `.expect()`, `.unwrap()`, and `assert_eq!`.
- **Recommendation**: Implement the requested error macro and replace panics/unwraps with cold-path error returns to improve branch prediction and code size in the hot path.

---

## **5. Extensibility**

### **Hardcoded Symbol Kinds**
- **Issue**: `SymbolKind` (defined in `dag`) is a closed enum.
- **Observation**: `ast_to_dag` has a large `match` statement on `SymbolKind`. This makes adding new primitive operations or custom JIT types difficult.
- **Recommendation**: Consider a trait-based or registration-based approach for custom operators if extensibility is a priority (as implied by the "custom rules" section of the plan).
