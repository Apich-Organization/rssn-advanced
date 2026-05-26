/*
 * basics.c — rssn-advanced C API walkthrough
 *
 * Demonstrates:
 *   1. Building a DAG expression (x^2 + 2*x + 1)
 *   2. Simplifying it with the heuristic engine
 *   3. JIT-compiling and evaluating it
 *   4. Registering a custom operator (relu) and evaluating relu(x+3)
 *
 * Build:  make -C examples all
 * Run:    make -C examples run
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../rssn-advanced.h"

/* ── helpers ──────────────────────────────────────────────────────────────── */

#define CHECK(call, msg)                                        \
    do {                                                        \
        enum RssnStatus _s = (call);                           \
        if (_s != RssnStatusSuccess) {                          \
            fprintf(stderr, "FAIL [status=%d]: %s\n", _s, msg); \
            exit(1);                                            \
        }                                                       \
    } while (0)

/* Simple relu — must have C linkage so the JIT can emit a direct call. */
static double relu_impl(double x) { return x > 0.0 ? x : 0.0; }

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    /* ── 1. Build the expression: x^2 + 2*x + 1 ─────────────────────────── */
    struct DagBuilder *dag = rssn_dag_new();
    if (!dag) { fputs("rssn_dag_new failed\n", stderr); return 1; }

    uint32_t x_id, two_id, one_id;
    uint32_t x2_id, two_x_id, sum1_id, root_id;

    CHECK(rssn_dag_variable_v2(dag, "x", &x_id),       "variable x");
    CHECK(rssn_dag_constant_v2(dag, 2.0, &two_id),     "constant 2");
    CHECK(rssn_dag_constant_v2(dag, 1.0, &one_id),     "constant 1");
    CHECK(rssn_dag_pow_v2    (dag, x_id, two_id, &x2_id),    "x^2");
    CHECK(rssn_dag_mul_v2    (dag, two_id, x_id, &two_x_id), "2*x");
    CHECK(rssn_dag_add_v2    (dag, x2_id, two_x_id, &sum1_id), "x^2+2x");
    CHECK(rssn_dag_add_v2    (dag, sum1_id, one_id, &root_id), "x^2+2x+1");

    printf("Expression built.  Root node id = %u\n", root_id);

    /* ── 2. JIT compile ──────────────────────────────────────────────────── */
    void *fn_ptr = NULL;
    CHECK(rssn_dag_compile_v2(dag, root_id, &fn_ptr), "compile x^2+2x+1");

    /* ── 3. Evaluate at several x values ────────────────────────────────── */
    printf("\nx^2 + 2x + 1  (should equal (x+1)^2):\n");
    double xs[] = { -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
    for (int i = 0; i < 6; i++) {
        double result = 0.0;
        double vars[] = { xs[i] };
        CHECK(rssn_dag_execute_v2(fn_ptr, vars, &result), "execute");
        printf("  f(%.1f) = %.1f   expected %.1f%s\n",
               xs[i], result, (xs[i] + 1.0) * (xs[i] + 1.0),
               result == (xs[i]+1.0)*(xs[i]+1.0) ? " ✓" : " ✗");
    }

    /* ── 4. Custom operator: relu ────────────────────────────────────────── */
    printf("\nCustom operator  relu(x + 3):\n");

    struct RssnCustomOpRegistry *reg = rssn_custom_op_registry_new();
    if (!reg) { fputs("registry_new failed\n", stderr); return 1; }

    /* Build relu(x + 3) first so we can intern the function name and get
     * the stable FnId assigned by the DagBuilder.  That same FnId must be
     * used when registering the eval_fn pointer in the custom-op registry. */
    struct DagBuilder *dag2 = rssn_dag_new();
    uint32_t x2, three_id, xp3_id, relu_fn_id, relu_node;
    CHECK(rssn_dag_variable_v2(dag2, "x",    &x2),       "variable x");
    CHECK(rssn_dag_constant_v2(dag2, 3.0,    &three_id), "constant 3");
    CHECK(rssn_dag_add_v2     (dag2, x2, three_id, &xp3_id), "x+3");

    /* Intern "relu" in dag2 — this allocates the FnId we must use everywhere. */
    relu_fn_id = rssn_dag_intern_function(dag2, "relu");
    if (relu_fn_id == (uint32_t)-1) {
        fputs("rssn_dag_intern_function failed\n", stderr); return 1;
    }

    /* Register the eval pointer under the SAME FnId that the builder assigned. */
    CHECK(rssn_custom_op_register_fn1(reg, relu_fn_id, "relu",
                                      (double(*)(double))relu_impl,
                                      /*vectorizable=*/1),
          "register relu");

    CHECK(rssn_dag_call_fn_v2 (dag2, relu_fn_id, &xp3_id, 1, &relu_node),
          "relu(x+3)");

    void *relu_fn = NULL;
    CHECK(rssn_dag_compile_with_custom_ops(dag2, relu_node, reg, &relu_fn),
          "compile relu(x+3)");

    double test_xs[] = { -5.0, -3.0, -1.0, 0.0, 2.0 };
    for (int i = 0; i < 5; i++) {
        double result = 0.0, expected;
        double v[] = { test_xs[i] };
        CHECK(rssn_dag_execute_v2(relu_fn, v, &result), "execute relu");
        expected = (test_xs[i] + 3.0) > 0.0 ? (test_xs[i] + 3.0) : 0.0;
        printf("  relu(%.1f + 3) = %.1f   expected %.1f%s\n",
               test_xs[i], result, expected,
               result == expected ? " ✓" : " ✗");
    }

    /* ── cleanup ─────────────────────────────────────────────────────────── */
    rssn_custom_op_registry_free(reg);
    rssn_dag_free(dag2);
    rssn_dag_free(dag);

    puts("\nDone.");
    return 0;
}
