# ============================================================
# check_snap_lin.jl -- standalone correctness check for STADE's
# snap_* (snap_plan) and lin_* (lin_build) against parse_kernel /
# shape_infer / act_analyze's real output.
#
# Structural checks only -- nothing here executes generated code
# (there isn't any yet; that's subtask 3). Run with:
#     julia check_snap_lin.jl
# from a directory containing STADE.jl and all_b.jl.
#
# Four groups:
#   1. snap_plan / :array,:value sites -- relu_field (a plain,
#      non-self-referencing write inside a *non*-sequential loop)
#      needs zero; geomrecur (a self-referencing recurrence inside a
#      sequential loop) needs exactly one, correctly identifying the
#      array; stencil_loss and cond_field_choice (writes that are
#      read again later, but never re-overwritten, and/or live in a
#      non-sequential loop) need zero -- confirming the check isn't
#      just "the array gets read again somewhere".
#   2. snap_plan / :branch sites -- exactly one per `if`, regardless
#      of nesting (branchsel: no loop; clamped_sumsq: if nested in a
#      loop; cond_field_choice: if wrapping a loop).
#   3. snap_plan / :tripcount sites -- a hand-written kernel with one
#      loop bound later reassigned (needs a site) and one that never
#      is (doesn't), isolating the rule from the noise of a real
#      multi-level kernel.
#   4. lin_build -- quadloss's single assignment: top-level op
#      matches the primal's own outermost operator (compared against
#      the parsed AST itself, not a hardcoded guess about how Julia
#      parses the nested expression), the rebuilt tree's `.expr`
#      reproduces the original rhs exactly (recursively, not just at
#      the root), and every leaf's activity matches active_map.
#      Also: lin_build's statement-list structure mirrors the
#      primal's own for/if nesting (relu_field).
# ============================================================

isdefined(Main, :lin_build) || include(joinpath(@__DIR__, "STADE.jl"))

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

# build (kernel, active_map) the same way stade_adjoint/stade_tangent
# do, with default (auto-derived) independents/dependents
function build(name::Symbol)
    kernel = parse_kernel(grab_kernel_expr(ALL_B_PATH, name))
    return kernel, act_analyze(kernel)
end

push_kinds(sites) = filter(s -> s.kind in (:array, :value), sites)
branch_sites(sites) = filter(s -> s.kind == :branch, sites)
tripcount_sites(sites) = filter(s -> s.kind == :tripcount, sites)

# ---- group 1: :array / :value sites ---------------------------------

println("--- group 1: :array / :value push sites ---")

let
    kernel, active_map = build(:relu_field)
    sites = snap_plan(kernel, active_map)
    @assert isempty(push_kinds(sites)) "relu_field: expected zero :array/:value sites, got $(push_kinds(sites))"
    println("relu_field           ok  (0 push sites -- v is never overwritten)")
end

let
    kernel, active_map = build(:geomrecur)
    sites = snap_plan(kernel, active_map)
    sitesp = push_kinds(sites)
    @assert length(sitesp) == 1 "geomrecur: expected exactly 1 :array/:value site, got $(length(sitesp))"
    @assert sitesp[1].kind == :array
    @assert sitesp[1].array == :u "geomrecur: push site should be for :u, got :$(sitesp[1].array)"
    println("geomrecur            ok  (1 push site, array = :u)")
end

let
    kernel, active_map = build(:stencil_loss)
    sites = snap_plan(kernel, active_map)
    @assert isempty(push_kinds(sites)) "stencil_loss: expected zero :array/:value sites, got $(push_kinds(sites))"
    println("stencil_loss         ok  (0 push sites -- w written once, u read-only)")
end

let
    kernel, active_map = build(:cond_field_choice)
    sites = snap_plan(kernel, active_map)
    @assert isempty(push_kinds(sites)) "cond_field_choice: expected zero :array/:value sites, got $(push_kinds(sites))"
    println("cond_field_choice    ok  (0 push sites -- w written once per branch, loop not sequential)")
end

# ---- group 2: :branch sites -------------------------------------------

println("--- group 2: :branch sites (one per `if`, regardless of nesting) ---")

for (name, expected) in [(:branchsel, 1), (:clamped_sumsq, 1), (:cond_field_choice, 1), (:relu_field, 1)]
    kernel, active_map = build(name)
    sites = snap_plan(kernel, active_map)
    got = length(branch_sites(sites))
    @assert got == expected "$name: expected $expected :branch site(s), got $got"
    println(rpad(String(name), 20), "ok  ($got branch site(s))")
end

# ---- group 3: :tripcount sites -----------------------------------------
# `n` bounds the first loop and is reassigned afterward -> needs a
# site; `m` bounds the second loop and is never reassigned -> doesn't.

tripcount_expr = :(function tripcount_kernel(x, out, n, m)
    for i = 1:n
        out[i] = x[i]
    end
    n = m
    for j = 1:m
        out[j] = out[j] + x[j]
    end
    return nothing
end)

println("--- group 3: :tripcount sites ---")
let
    kernel = parse_kernel(tripcount_expr)
    active_map = act_analyze(kernel)
    sites = snap_plan(kernel, active_map)
    trips = tripcount_sites(sites)
    @assert length(trips) == 1 "tripcount_kernel: expected exactly 1 :tripcount site, got $(length(trips))"
    @assert trips[1].array == :n "tripcount_kernel: expected the site to name :n, got :$(trips[1].array)"
    println("tripcount_kernel     ok  (1 tripcount site, array = :n; :m correctly excluded)")
end

# ---- group 4: lin_build ------------------------------------------------

println("--- group 4: lin_build ---")

# recursively collect every :leaf node of a lin_node tree
function collect_leaves(node, out)
    if node.kind == :leaf
        push!(out, node)
    else
        for c in node.children
            collect_leaves(c, out)
        end
    end
    return out
end

let
    kernel, active_map = build(:quadloss)
    plan = lin_build(kernel, active_map)
    @assert length(plan) == 1
    @assert plan[1].kind == :assign
    assign_stmt = kernel.body[1]
    @assert assign_stmt.kind == :assign

    tree = plan[1].tree
    expected_op = assign_stmt.rhs.args[1]   # derived from the parsed AST, not assumed
    @assert tree.kind == :op
    @assert tree.op == expected_op "quadloss: top-level op $(tree.op) != primal's outermost operator $expected_op"

    # the rebuilt primal expression must reproduce the original rhs
    # exactly, recursively -- not just match at the root
    @assert tree.expr == assign_stmt.rhs "quadloss: rebuilt tree.expr does not reproduce the primal rhs"

    # x, y, z are all independents here (every float arg) -> root is active
    @assert tree.active == true
    @assert plan[1].active == true

    # every leaf's activity matches active_map / literal-ness
    leaves = collect_leaves(tree, NamedTuple[])
    @assert length(leaves) >= 3   # at least x, y, z should appear as leaves somewhere
    for leaf in leaves
        if leaf.expr isa Symbol
            @assert leaf.active == active_map[leaf.expr] "quadloss: leaf :$(leaf.expr) activity mismatch"
        elseif leaf.expr isa Number
            @assert leaf.active == false "quadloss: literal $(leaf.expr) should never be active"
        end
    end
    println("quadloss             ok  (top-level op = :$expected_op, tree.expr reproduces rhs, leaf activity correct)")
end

# a second data point for the "tree.expr reproduces rhs exactly"
# check, on a kernel with array-ref leaves and a 3-term sum instead
# of quadloss's mixed +/-/^ shape
let
    kernel, active_map = build(:stencil_loss)
    plan = lin_build(kernel, active_map)
    # stencil_loss's body is `for {assign}; for {assign}` -- the `w`
    # computation is nested one level inside the first loop
    @assert plan[1].kind == :for
    tree = plan[1].body[1].tree
    primal_rhs = kernel.body[1].body[1].rhs
    @assert tree.expr == primal_rhs "stencil_loss: rebuilt tree.expr does not reproduce the primal rhs"
    @assert tree.active == true   # u is independent
    println("stencil_loss         ok  (tree.expr reproduces rhs)")
end

# lin_build's statement-list structure must mirror the primal's own
# for/if nesting -- relu_field is `for { if {assign} else {assign} }`
# followed by a second top-level `for { assign }`
let
    kernel, active_map = build(:relu_field)
    plan = lin_build(kernel, active_map)
    @assert length(plan) == length(kernel.body) == 2
    @assert plan[1].kind == :for
    @assert plan[2].kind == :for
    inner = plan[1].body
    @assert length(inner) == 1
    @assert inner[1].kind == :if
    @assert length(inner[1].then) == 1 && inner[1].then[1].kind == :assign
    @assert length(inner[1].els) == 1 && inner[1].els[1].kind == :assign
    println("relu_field           ok  (lin_plan structure mirrors primal for/if nesting)")
end

# ---- bonus robustness: mg_vcycle (M4) doesn't crash, and `at` is a
# dense 1:length(sites) sequence in emission order ------------------

println("--- bonus: mg_vcycle robustness (deep nesting, many sites) ---")
let
    kernel, active_map = build(:mg_vcycle)
    sites = snap_plan(kernel, active_map)
    plan = lin_build(kernel, active_map)
    @assert !isempty(sites)
    @assert [s.at for s in sites] == collect(1:length(sites)) "at-values are not a dense, ordered 1:N sequence"
    @assert length(plan) == length(kernel.body)
    println("mg_vcycle            ok  ($(length(sites)) total sites, at-values dense & ordered)")
end

println("check_snap_lin.jl: all snap_* / lin_* checks passed")