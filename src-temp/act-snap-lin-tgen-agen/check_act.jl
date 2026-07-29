# ============================================================
# check_act.jl -- standalone correctness check for STADE's act_*
# (act_analyze / act_propagate! / act_propagate_assign! /
# act_expr_active) against parse_kernel/shape_infer's real output.
#
# Not part of STADE.jl itself -- a throwaway harness, same pattern as
# check_parse_m1.jl / check_der.jl. Run with:
#     julia check_act.jl
# from a directory containing STADE.jl and all_b.jl.
#
# Three groups of checks:
#   1. corpus invariant: for a spread of real M1-M4 kernels, every
#      float-kinded arg ends up active and every Int64-kinded
#      variable (arg or local -- loop counters, branch selectors,
#      sizes) never does, regardless of nesting depth.
#   2. hand-written kernels isolating the two properties the corpus
#      invariant alone can't exercise (every corpus kernel's floats
#      are auto-independent from the start, so genuine propagation
#      and multi-pass convergence never actually get triggered by
#      test group 1):
#        - monotonicity: a variable tainted by one statement stays
#          active even after a later statement fully overwrites it
#          with something inactive.
#        - fixed-point necessity: taint that a single top-down pass
#          over one loop body can't see (an earlier statement reads
#          a variable a later statement in the same body activates)
#          is still picked up because act_analyze re-sweeps to a
#          fixed point -- demonstrated by contrasting a single
#          act_propagate! pass (which must NOT converge) against the
#          full act_analyze (which must).
#   3. if/else union: a variable becomes active if EITHER branch of
#      an `if` could taint it, since activity is a static
#      may-reach property, not tied to which branch runs at runtime.
# ============================================================

isdefined(Main, :act_analyze) || include(joinpath(@__DIR__, "STADE.jl"))

# pull `function name(...) ... end` back out of all_b.jl as a raw
# Expr, the same way io_read_kernel would from a single-function file
function grab_kernel_expr(path::String, name::Symbol)
    parsed = Meta.parseall(read(path, String))
    for e in parsed.args
        if e isa Expr && e.head == :function
            sig = e.args[1]
            if sig isa Expr && sig.head == :call && sig.args[1] == name
                return e
            end
        end
    end
    error("no `function $name(...)` found in $path")
end

const ALL_B_PATH = joinpath(@__DIR__, "all_b.jl")

# ---- group 1: corpus invariant across M1-M4 ------------------------

corpus_names = [
    :dotprod, :quadloss, :bilinear,          # M1
    :branchsel, :clamped_sumsq, :cond_field_choice,  # M2
    :geomrecur, :stencil_loss, :func,        # M3
    :mg_vcycle,                              # M4
]

println("--- group 1: corpus invariant (float args active, Int64 vars never) ---")
for name in corpus_names
    expr = grab_kernel_expr(ALL_B_PATH, name)
    kernel = parse_kernel(expr)
    active_map = act_analyze(kernel)

    # act_analyze covers exactly the variables shape_infer found --
    # no var silently dropped, none invented
    @assert Set(keys(active_map)) == Set(keys(kernel.sig.kinds)) "$name: active_map/kinds key mismatch"

    for a in kernel.sig.args
        kind = kernel.sig.kinds[a]
        if kind in (:scalar_float, :array_float)
            @assert active_map[a] == true "$name: float arg :$a expected active, got inactive"
        else
            @assert active_map[a] == false "$name: Int64 arg :$a expected inactive, got active"
        end
    end

    # global invariant: no Int64-kinded variable anywhere (arg or
    # local -- loop counters, branch selectors, sizes) is ever active
    for (v, kind) in kernel.sig.kinds
        if kind in (:scalar_int, :array_int)
            @assert active_map[v] == false "$name: Int64 var :$v ($kind) marked active"
        end
    end
    println(rpad(String(name), 18), "ok  (", length(active_map), " vars)")
end

# ---- group 2a: monotonicity -- tainted-then-overwritten scratch array ----
# scratch is excluded from the auto-derived independents so the only
# way it can end up active is genuine propagation from x, not from
# being a float arg by default.

scratch_expr = :(function scratch_kernel(x, scratch, n)
    for i = 1:n
        scratch[i] = x[i]
    end
    for i = 1:n
        scratch[i] = 0.0
    end
    return nothing
end)

println("--- group 2a: monotonicity (scratch array tainted, then fully overwritten) ---")
let
    kernel = parse_kernel(scratch_expr)
    kernel = parse_override_indep_dep(kernel, [:x], [:x])
    active_map = act_analyze(kernel)
    @assert active_map[:x] == true
    @assert active_map[:scratch] == true   # tainted by the first loop
    @assert active_map[:n] == false        # loop bound, Int64
    @assert active_map[:i] == false        # loop counter, Int64
    println("scratch_kernel      ok  (scratch active despite later `scratch[i] = 0.0`)")
end

# ---- group 2b: fixed-point necessity --------------------------------
# b[i] = a[i] appears (textually) BEFORE a[i] = x[i] inside the same
# loop body. A single top-down pass over the body activates `a` (from
# x) but never revisits the `b` statement to notice it -- only a
# second pass over the whole body does. act_analyze must converge to
# b active; a single act_propagate! call must NOT.

fixedpoint_expr = :(function fixedpoint_kernel(x, a, b, n)
    for i = 1:n
        b[i] = a[i]
        a[i] = x[i]
    end
    return nothing
end)

println("--- group 2b: fixed-point necessity (multi-pass convergence within one loop body) ---")
let
    kernel = parse_kernel(fixedpoint_expr)
    kernel = parse_override_indep_dep(kernel, [:x], [:x])

    # a single pass: a becomes active (from x), but b does NOT --
    # this is the regression guard that act_analyze isn't secretly
    # just doing one act_propagate! call
    one_pass_map = Dict{Symbol,Bool}(v => false for v in keys(kernel.sig.kinds))
    one_pass_map[:x] = true
    act_propagate!(kernel.body, one_pass_map, kernel.sig.kinds)
    @assert one_pass_map[:a] == true
    @assert one_pass_map[:b] == false "a single act_propagate! pass unexpectedly converged -- test no longer distinguishes single-pass from fixed-point"

    # the real, full act_analyze must converge all the way
    active_map = act_analyze(kernel)
    @assert active_map[:x] == true
    @assert active_map[:a] == true
    @assert active_map[:b] == true "act_analyze failed to reach the fixed point: b should be active via a second pass over the loop body"
    println("fixedpoint_kernel   ok  (b active only after >1 pass, as expected)")
end

# ---- group 3: if/else union ------------------------------------------
# out1 is tainted only in the `then` arm, out2 only in the `else` arm
# -- both must end up active, since activity is a static "might this
# ever carry a derivative" property across both arms, not a trace of
# whichever branch happens to run.

ifmerge_expr = :(function ifmerge_kernel(x, out1, out2, i_branch)
    if i_branch == 1
        out1 = x
    else
        out1 = 2.0
    end
    if i_branch == 1
        out2 = 3.0
    else
        out2 = x
    end
    return nothing
end)

println("--- group 3: if/else union (either arm can taint the merge) ---")
let
    kernel = parse_kernel(ifmerge_expr)
    kernel = parse_override_indep_dep(kernel, [:x], [:x])
    active_map = act_analyze(kernel)
    @assert active_map[:x] == true
    @assert active_map[:out1] == true "out1 should be active via the `then` arm (`out1 = x`)"
    @assert active_map[:out2] == true "out2 should be active via the `else` arm (`out2 = x`)"
    @assert active_map[:i_branch] == false   # Int64 selector
    println("ifmerge_kernel      ok  (both out1 and out2 active via opposite arms)")
end

println("check_act.jl: all act_* checks passed")