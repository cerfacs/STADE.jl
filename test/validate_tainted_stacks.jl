# Regression for v0.5.15: stacks that fall back to push!/pop! inside an indexed (keep_push_pop = false) adjoint.
#   Defect 1 (phc_mg_solve): an `if` whose arm pushes more entries on the same shared flag stack. In indexed mode the flag was pushed BEFORE the arm,
#   so the backward `pop!` of the flag read an entry of the arm: a silently wrong gradient for one pass and a BoundsError for two or more.
#   keep_push_pop = true already pushed the flag AFTER the arm. Fix: a flag stack in `tainted_stacks` (push!/pop! fallback) follows that order.
#   Defect 3: a scalar that the guard arm alone assigns (`nfm = nfine - 1`) was read by the sizing loop of initstacks in a pass where the guard is false:
#   UndefVarError. The scalar now starts at 0 before that loop.
#   Defect 2: initstacks counted its slots with `x += 1`, which cgen_parse_generated does not read, so the GPU ingest of the initstacks refused it.
#   Defect 4: the host code pushed a device element onto a push!/pop! stack with no allowscalar block (`push!(u_stack, u[j, l])`): "Scalar indexing is
#   disallowed" on the V100 for phc_mg_solve, CUDA and JACC.
# usage: julia validate_tainted_stacks.jl
include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

const TS_SRC = """
function ts_guard_nest(u, f, nfine, maxit)
    rn = 1.0
    for kk = 1:maxit
        if rn > 0.0
            nl = nfine
            for lv = 1:3
                n = nl - 1
                for j = 1:n
                    left = 0.0
                    if j > 1
                        left = u[j - 1]
                    end
                    u[j] = 0.5 * u[j] + 0.1 * left + f[j]
                end
                nl = div(nl, 2)
            end
            rn = 0.0
            for j = 1:3
                rn = rn + u[j] * u[j]
            end
        end
    end
    return nothing
end"""

# the then-arm of the first `if` that holds a flag push: is the flag push its LAST statement?
function ts_flag_last(e)
    e isa Expr || return nothing
    if e.head in (:if, :elseif) && length(e.args) >= 2 && e.args[2] isa Expr && e.args[2].head == :block
        st = [s for s in e.args[2].args if !(s isa LineNumberNode)]
        isflag(s) = s isa Expr && s.head == :call && s.args[1] == :push! && s.args[2] == :branch_stack
        if any(isflag, st) && any(s -> s isa Expr && s.head == :for, st)
            return isflag(st[end])
        end
    end
    for a in e.args
        r = ts_flag_last(a); r === nothing || return r
    end
    return nothing
end

# the same statement list with the flag push moved to the top of the arm: what the old code emitted
function ts_flag_first(e)
    e isa Expr || return e
    args = Any[ts_flag_first(a) for a in e.args]
    if e.head in (:if, :elseif) && length(args) >= 2 && args[2] isa Expr && args[2].head == :block
        st = [s for s in args[2].args if !(s isa LineNumberNode)]
        isflag(s) = s isa Expr && s.head == :call && s.args[1] == :push! && s.args[2] == :branch_stack
        if length(st) > 1 && isflag(st[end]) && any(s -> s isa Expr && s.head == :for, st)
            args[2] = Expr(:block, st[end], st[1:end-1]...)
        end
    end
    return Expr(e.head, args...)
end

const TS_SRC2 = """
function ts_guard_scalar(u, f, nfine, maxit)
    rn = 1.0
    for kk = 1:maxit
        if rn > 0.5
            nl = nfine
            nfm = nfine - 1
            for lv = 1:3
                n = nl - 1
                for j = 1:n
                    u[j] = 0.5 * u[j] + f[j]
                end
                nl = div(nl, 2)
            end
            rn = 0.0
            for j = 1:nfm
                rn = rn + u[j] * u[j]
            end
        end
    end
    return nothing
end"""

# number of `push!(stack, <expression with an array read>)` outside an allowscalar macro call
function ts_bare_push(e, inside = false)
    e isa Expr || return 0
    if e.head == :macrocall
        nm = string(e.args[1])
        return sum(ts_bare_push(a, inside || occursin("allowscalar", nm)) for a in e.args[2:end]; init = 0)
    end
    n = 0
    if e.head == :call && length(e.args) == 3 && e.args[1] == :push! && !inside && STADE.cgen_expr_has_ref(e.args[3])
        n += 1
    end
    return n + sum(ts_bare_push(a, inside) for a in e.args; init = 0)
end

function validate_tainted_stacks(; verbose = true)
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    e = Meta.parse(TS_SRC); k = STADE.parse_kernel(e)
    a0 = STADE.stade_adjoint(e; keep_push_pop = true); a1 = STADE.stade_adjoint(e; keep_push_pop = false)
    # the case under test must be the fallback one
    say(occursin("push!(branch_stack", string(a1.adjoint)), "the indexed adjoint no longer uses a push!/pop! flag stack: the case is not exercised")
    # order: the flag is the last statement of the arm, and the detector sees the old order
    say(ts_flag_last(a1.adjoint) === true, "flag push is not the last statement of the guard arm")
    say(ts_flag_last(a0.adjoint) === true, "stack mode: flag push is not the last statement of the guard arm")
    say(ts_flag_last(ts_flag_first(a1.adjoint)) === false, "sabotage: detector missed the flag at the top of the arm")
    # gradient: indexed against stack mode, one to three passes
    f0, i0 = STADE.val_compile(a0.adjoint), STADE.val_compile(a0.initstacks)
    f1, i1 = STADE.val_compile(a1.adjoint), STADE.val_compile(a1.initstacks)
    san = Symbol[a for a in a1.initstacks.args[1].args[2:end]]
    for maxit in 1:3
        I = Dict(:nfine => 16, :maxit => maxit)
        V = Dict{Symbol,Any}(:u => 0.1 .* collect(1.0:15.0), :f => 0.01 .* collect(1.0:15.0))
        seed = Dict{Symbol,Any}(:u => ones(15), :f => zeros(15))
        g0 = STADE.val_call_adjoint(f0, i0, k, I, deepcopy(V), seed)
        g1 = try STADE.val_call_adjoint(f1, i1, k, I, deepcopy(V), seed; stack_arg_names = san) catch ex; first(sprint(showerror, ex), 80) end
        say(g1 isa AbstractVector && maximum(abs.(g0 .- g1)) < 1e-12, "maxit=$maxit: indexed gradient differs from stack mode ($(g1 isa AbstractVector ? maximum(abs.(g0 .- g1)) : g1))")
    end
    # a scalar assigned only inside the guard: the guard is true in pass 1 and false afterwards
    e2 = Meta.parse(TS_SRC2); k2 = STADE.parse_kernel(e2)
    b0 = STADE.stade_adjoint(e2; keep_push_pop = true); b1 = STADE.stade_adjoint(e2; keep_push_pop = false)
    g0f, g0i = STADE.val_compile(b0.adjoint), STADE.val_compile(b0.initstacks)
    g1f, g1i = STADE.val_compile(b1.adjoint), STADE.val_compile(b1.initstacks)
    san2 = Symbol[a for a in b1.initstacks.args[1].args[2:end]]
    for maxit in 1:4
        I = Dict(:nfine => 16, :maxit => maxit)
        V = Dict{Symbol,Any}(:u => 0.01 .* collect(1.0:15.0), :f => 0.001 .* collect(1.0:15.0))
        seed = Dict{Symbol,Any}(:u => ones(15), :f => zeros(15))
        g0 = STADE.val_call_adjoint(g0f, g0i, k2, I, deepcopy(V), seed)
        g1 = try STADE.val_call_adjoint(g1f, g1i, k2, I, deepcopy(V), seed; stack_arg_names = san2) catch ex; first(sprint(showerror, ex), 80) end
        say(g1 isa AbstractVector && maximum(abs.(g0 .- g1)) < 1e-12, "scalar in guard, maxit=$maxit: $(g1 isa AbstractVector ? maximum(abs.(g0 .- g1)) : g1)")
    end
    # initstacks: no `+=` counters, and the GPU ingest accepts it (CUDA and JACC), adjoint and HVP
    hasplus(x) = x isa Expr && (x.head == :(+=) || any(hasplus, x.args))
    h1 = STADE.stade_hvp(e; keep_push_pop = false)
    for (mode, gen) in (("adjoint", a1), ("hvp", h1))
        say(!hasplus(gen.initstacks), "$mode: initstacks still has a `+=` counter")
        for (label, f) in (("cuda", () -> STADE.stade_gpu(gen.initstacks, STADE.cgen_backend_cuda())),
                           ("jacc", () -> STADE.stade_jacc(gen.initstacks)))
            ok = try f(); true catch ex; verbose && println("  ", label, " ", mode, ": ", first(sprint(showerror, ex), 160)); false end
            say(ok, "$mode/$label: the initstacks is refused by the GPU ingest")
        end
    end
    # host push! of a device element: wrapped, CUDA and JACC, adjoint and HVP (cgen_host_box_refs-style scan of the generated host code)
    for (mode, gen) in (("adjoint", a1), ("hvp", h1))
        ex = mode == "hvp" ? gen.hvp : gen.adjoint
        hc = STADE.stade_gpu(ex, STADE.cgen_backend_cuda()).host
        hj = STADE.stade_jacc(ex).host
        say(occursin("push!", string(hc)) && occursin("push!", string(hj)), "$mode: no host push! to scan")
        say(ts_bare_push(hc) == 0, "$mode/cuda: $(ts_bare_push(hc)) push! of a device element outside allowscalar")
        say(ts_bare_push(hj) == 0, "$mode/jacc: $(ts_bare_push(hj)) push! of a device element outside allowscalar")
        # sabotage: the detector must see a bare push
        say(ts_bare_push(Meta.parse("begin; push!(s, u[1]); end")) == 1, "sabotage: detector missed a bare push of a device element")
    end
    println(bad == 0 ? "all $checks tainted-stack checks passed" : "$bad of $checks NOT OK")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_tainted_stacks() == 0 ? 0 : 1)
end
