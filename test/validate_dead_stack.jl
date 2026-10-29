# ==================== validate_dead_stack.jl =====================
# Checks the deletion of dead stack stores (`dead_stack`, v0.4.44). A forward kernel pushes a running sum to a stack at every step; the backward kernel reads one
# site back. The stores of the sites no read repeats are dead, and `dead_stack = 1` deletes them.
#
#   julia validate_dead_stack.jl           units, refusals, sabotage
#   validate_dead_stack(; phases = (:units,))
#
# Every case is a layer in the form the adjoint stage generates (`ds_src`), with one edit. It runs through the CPU shim (ownership_shim.jl) twice from the same random
# inputs, with `dead_stack = 1` and with `dead_stack = 0`, and once on the CPU as the source function.
#   - a case that must delete: the forward kernel has fewer stores to the stack, the outputs other than the stack are the same numbers as with `dead_stack = 0`, they
#     agree with the CPU run to 1e-12, the shim reports no conflict, and outputs changed;
#   - a case that must refuse: the kernel text of both plans is the same for that stack. The other stack (`t_stack`, never read) is stripped in every case, so a refusal
#     is not a plan that deleted nothing at all.
# SABOTAGE edits of the pass, each of which one case must catch.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
isdefined(Main, :con_init!) || include(joinpath(@__DIR__, "validate_contraction.jl"))
using Random

# a layer of m units on n inputs, in the form of the adjoint stage. `s_stack` has three sites: the sum at every input (region 1), the final sum (region 2, read back by
# the backward kernel). `t_stack` is written and never read.
const DS_BASE = """function ds_f(x, xb, w, wb, b, bb, y, yb, s_stack, t_stack, n, m)
    s = 0.0
    sb = 0.0
    for j = 1:m
        s = b[j]
        for i = 1:n
            s = s + w[(i - 1) * m + j] * x[i]
            __idx_s_stack_1 = ((j - 1) + (i - 1) * m) + 1
            s_stack[__idx_s_stack_1] = s
        end
        __idx_s_stack_2 = m * n + ((j - 1) + 1)
        s_stack[__idx_s_stack_2] = s
        __idx_t_stack_1 = (j - 1) + 1
        t_stack[__idx_t_stack_1] = s * 2.0
        y[j] = max(s, 0.0)
    end
    for j = m:-1:1
        __idx_s_stack_0 = m * n + ((j - 1) + 1)
        s = s_stack[__idx_s_stack_0]
        __oldb_0 = yb[j]
        yb[j] = 0.0
        sb = (0.5 * (1.0 + sign(s))) * __oldb_0
        for i = n:-1:1
            wb[(i - 1) * m + j] = wb[(i - 1) * m + j] + x[i] * sb
            xb[i] = xb[i] + w[(i - 1) * m + j] * sb
        end
        bb[j] = bb[j] + sb
    end
    return nothing
end
"""

function ds_edit(src::String, old::String, new::String; nth::Int = 1)
    r = findall(old, src)
    length(r) >= nth || error("ds_edit: anchor not found: $old")
    return src[1:first(r[nth])-1] * new * src[last(r[nth])+1:end]
end

# n = 4 inputs, m = 5 units; the stack holds the two regions
function ds_args(sd)
    r = MersenneTwister(sd)
    n, m = 4, 5
    R(k) = randn(r, k)
    return Any[R(n), R(n), R(n * m), R(n * m), R(m), R(m), R(m), R(m), zeros(m * n + m), zeros(m), n, m]
end

function ds_run(src::String, args; dead::Int = 1, run::Bool = true, init::Bool = true, initsrc::String = "")
    con_init!()
    return Base.invokelatest(ds_run_body, src, args, dead, run, init, initsrc)
end
function ds_run_body(src::String, args, dead::Int, run::Bool, init::Bool, initsrc::String)
    Main.CUDA.MAXT[] = name -> 1024
    cuda = STADE.cgen_backend_cuda()
    kw = (warp_atomics = false, shared_private = false, split_reduce = 0, contraction = 0, reuse_stored = 0, dead_stack = dead)
    # the init host is what tells the pass that the arrays are STADE's own stacks
    pinit = STADE.stade_gpu(Meta.parse(isempty(initsrc) ? DS_INIT : initsrc), cuda; kw...)
    plan = STADE.stade_gpu(Meta.parse(src), cuda; kw..., init_hosts = init ? Dict{Symbol,Expr}(pinit.host.args[1].args[1] => pinit.host) : nothing)
    ktxt = Dict{String,String}(string(k.args[1].args[1]) => string(k) for k in plan.kernels if k.head === :function)
    run || return (outs = nothing, conflicts = Any[], plan = plan, kernels = ktxt)
    for k in plan.kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host)
    host = plan.host.args[1].args[1]
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS)
    Base.invokelatest(getfield(Main, host), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    return (outs = outs, conflicts = copy(Main.CUDA.VIOLATIONS), plan = plan, kernels = ktxt)
end

function ds_cpu(src::String, args)
    Core.eval(Main, Meta.parse(src))
    cp = Any[a isa AbstractArray ? copy(a) : a for a in args]
    Base.invokelatest(getfield(Main, :ds_f), cp...)
    return cp
end

# all arrays but the two stacks (positions 9 and 10)
ds_vis(outs) = [outs[i] for i in 1:8]
ds_same(a, b) = all(x == y for (x, y) in zip(ds_vis(a), ds_vis(b)))
ds_relerr(a, b) = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(ds_vis(a), ds_vis(b)); init = 0.0)
ds_changed(outs, args) = sum(count(x .!= y) for (x, y) in zip(ds_vis(outs), ds_vis(args)); init = 0)
ds_count(r, name) = sum(length(findall(name * "[", v)) for v in values(r.kernels); init = 0)

# (label, edit, stores of s_stack with dead_stack = 0, stores of s_stack with dead_stack = 1)
ds_cases() = Tuple[
    ("the backward kernel reads a second site (the sum at every input)",
        ds_edit(DS_BASE, "wb[(i - 1) * m + j] = wb[(i - 1) * m + j] + x[i] * sb",
                "__idx_s_stack_3 = ((j - 1) + (i - 1) * m) + 1\n            s2 = s_stack[__idx_s_stack_3]\n            wb[(i - 1) * m + j] = wb[(i - 1) * m + j] + x[i] * sb + 0.0 * s2"), 2, 2),
    ("a read at the same slots written another way",
        ds_edit(DS_BASE, "__idx_s_stack_0 = m * n + ((j - 1) + 1)", "__idx_s_stack_0 = ((j - 1) + 1) + m * n"), 2, 2),
    ("the read index goes through a local scalar",
        ds_edit(DS_BASE, "__idx_s_stack_0 = m * n + ((j - 1) + 1)", "jj = m * n + ((j - 1) + 1)\n        __idx_s_stack_0 = jj"), 2, 2),
    ("the read repeats the text of a store, a local scalar, at the slots of another site",
        ds_edit(ds_edit(DS_BASE, "__idx_s_stack_2 = m * n + ((j - 1) + 1)\n        s_stack[__idx_s_stack_2] = s", "jj = m * n + ((j - 1) + 1)\n        s_stack[jj] = s"),
                "__idx_s_stack_0 = m * n + ((j - 1) + 1)\n        s = s_stack[__idx_s_stack_0]", "jj = (j - 1) + (n - 1) * m + 1\n        s = s_stack[jj]"), 2, 2),
    ("a host scalar of the index changes between the launches",
        ds_edit(DS_BASE, "    for j = m:-1:1\n", "    m = m - 1\n    for j = m:-1:1\n"), 2, 2),
]


# --- the allocation: the kept region moves to the start of the array (dead_stack = 2)
const DS_INIT = """function initstacks_ds_f(n, m)
    s_stack = Vector{Float64}(undef, (m * n + m))
    t_stack = Vector{Float64}(undef, m)
    return (s_stack, t_stack)
end
"""

function ds_run_c(src::String, initsrc::String, args; level::Int = 2)
    con_init!()
    return Base.invokelatest(ds_run_c_body, src, initsrc, args, level)
end
function ds_run_c_body(src::String, initsrc::String, args, level::Int)
    Main.CUDA.MAXT[] = name -> 1024
    cuda = STADE.cgen_backend_cuda()
    kw = (warp_atomics = false, shared_private = false, split_reduce = 0, contraction = 0, reuse_stored = 0, dead_stack = level)
    pinit = STADE.stade_gpu(Meta.parse(initsrc), cuda; kw...)
    plan = STADE.stade_gpu(Meta.parse(src), cuda; kw..., init_hosts = Dict{Symbol,Expr}(pinit.host.args[1].args[1] => pinit.host))
    ihost = STADE.cgen_shrink_stacks(pinit.host, plan.dead_stacks, plan.compact_stacks)
    for k in plan.kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host); Core.eval(Main, ihost)
    stacks = Base.invokelatest(getfield(Main, ihost.args[1].args[1]), args[11], args[12])
    foreach(st -> fill!(st.data, 0.0), stacks)
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    dev[9] = stacks[1]; dev[10] = stacks[2]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS)
    Base.invokelatest(getfield(Main, plan.host.args[1].args[1]), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    ktxt = Dict{String,String}(string(k.args[1].args[1]) => string(k) for k in plan.kernels if k.head === :function)
    return (outs = outs, conflicts = copy(Main.CUDA.VIOLATIONS), plan = plan, kernels = ktxt, sizes = (length(stacks[1].data), length(stacks[2].data)))
end

function ds_compact_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    args = ds_args(1)
    c2 = ds_run_c(DS_BASE, DS_INIT, args; level = 2)
    c1 = ds_run_c(DS_BASE, DS_INIT, args; level = 1)
    c0 = ds_run_c(DS_BASE, DS_INIT, args; level = 0)
    cpu = ds_cpu(DS_BASE, args)
    check(c0.sizes == (25, 5) && c1.sizes == (25, 1) && c2.sizes == (5, 1), "compact: stack sizes are $(c0.sizes) with dead_stack = 0, $(c1.sizes) with 1, $(c2.sizes) with 2")
    check(haskey(c2.plan.compact_stacks, :s_stack) && !haskey(c1.plan.compact_stacks, :s_stack), "compact: plan.compact_stacks names s_stack only at level 2")
    check(all(occursin("s_stack[(j - 1) + 1]", v) || !occursin("s_stack[", v) for v in values(c2.kernels)) && ds_count(c2, "s_stack") == 2, "compact: both accesses of s_stack index by their own term")
    check(ds_same(c2.outs, c0.outs) && ds_same(c1.outs, c0.outs), "compact: the outputs are the same numbers at levels 0, 1 and 2")
    check(ds_relerr(c2.outs, cpu) <= 1e-12 && isempty(c2.conflicts), "compact: the outputs agree with the CPU run (error $(round(ds_relerr(c2.outs, cpu), sigdigits = 3))), no conflict")
    check(ds_changed(c2.outs, args) > 0, "compact: the run changed $(ds_changed(c2.outs, args)) elements")
    # a region after the kept one stays in the array: the size is the sum of the terms from the kept region on
    ce = ds_run_c(DS_BASE, replace(DS_INIT, "(m * n + m)" => "((m * n + m) + 3)"), args; level = 2)
    ce0 = ds_run_c(DS_BASE, replace(DS_INIT, "(m * n + m)" => "((m * n + m) + 3)"), args; level = 0)
    check(ce.sizes[1] == 5 + 3 && ds_same(ce.outs, ce0.outs), "compact: with a term after the kept region the array holds both ($(ce.sizes[1]) slots)")
    # refusals: every one keeps the size of the array, and the outputs are the same numbers
    refuse = [
        ("the allocation lists the regions in another order", replace(DS_INIT, "(m * n + m)" => "(m + m * n)"), DS_BASE),
        ("the allocation has another prefix", replace(DS_INIT, "(m * n + m)" => "((m + 1) + m * n)"), DS_BASE),
        ("the backward kernel reads two regions",
            ds_edit(DS_BASE, "wb[(i - 1) * m + j] = wb[(i - 1) * m + j] + x[i] * sb",
                    "__idx_s_stack_3 = ((j - 1) + (i - 1) * m) + 1\n            s2 = s_stack[__idx_s_stack_3]\n            wb[(i - 1) * m + j] = wb[(i - 1) * m + j] + x[i] * sb + 0.0 * s2"), nothing),
    ]
    for (label, init, src) in refuse
        init2 = init isa String && src !== nothing ? init : DS_INIT
        src2 = src === nothing ? init : src
        r = ds_run_c(src2, init2, args; level = 2)
        r0 = ds_run_c(src2, init2, args; level = 0)
        check(r.sizes[1] == r0.sizes[1] && isempty(r.plan.compact_stacks), "compact refused ($label): the array keeps its size $(r.sizes[1]) and nothing is reported")
        check(ds_same(r.outs, r0.outs), "compact refused ($label): the outputs are the same numbers")
    end
    return bad
end

function ds_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    args = ds_args(1)
    a = ds_run(DS_BASE, args; dead = 1)
    b = ds_run(DS_BASE, args; dead = 0)
    cpu = ds_cpu(DS_BASE, args)
    check(ds_count(b, "s_stack") == 3 && ds_count(a, "s_stack") == 2, "base: s_stack has $(ds_count(b, "s_stack")) accesses with dead_stack = 0 and $(ds_count(a, "s_stack")) with 1 (final store and read)")
    check(ds_count(b, "t_stack") == 1 && ds_count(a, "t_stack") == 0, "base: the store to t_stack is deleted ($(ds_count(b, "t_stack")) -> $(ds_count(a, "t_stack")))")
    check(a.plan.dead_stacks == [:t_stack], "base: plan.dead_stacks is [:t_stack] (got $(a.plan.dead_stacks))")
    check(b.plan.dead_stacks == Symbol[], "base: dead_stack = 0 reports no dead stack")
    check(ds_same(a.outs, b.outs), "base: the outputs other than the stacks are the same numbers with dead_stack = 0 and 1")
    check(ds_relerr(a.outs, cpu) <= 1e-12, "base: the outputs agree with the CPU run (error $(round(ds_relerr(a.outs, cpu), sigdigits = 3)))")
    check(isempty(a.conflicts), "base: the shim reports no conflict")
    check(ds_changed(a.outs, args) > 0, "base: the run changed $(ds_changed(a.outs, args)) elements")
    check(a.outs[9][end-4:end] == b.outs[9][end-4:end], "base: the final-site slots hold the same values")
    # without the init host the arrays are not known to be STADE's stacks (a caller's array ending in _stack may be an output): nothing is deleted
    n_ = ds_run(DS_BASE, args; dead = 2, init = false)
    check(ds_count(n_, "s_stack") == 3 && ds_count(n_, "t_stack") == 1 && isempty(n_.plan.dead_stacks) && ds_same(n_.outs, b.outs), "no init host: the stores to s_stack and t_stack stay (got $(ds_count(n_, "s_stack")) and $(ds_count(n_, "t_stack")))")
    # an init host that allocates s_stack only is no evidence for t_stack: that array may be the caller's
    p_ = ds_run(DS_BASE, args; dead = 2, initsrc = "function initstacks_ds_f(n, m)\n    s_stack = Vector{Float64}(undef, (m * n + m))\n    return (s_stack,)\nend")
    check(ds_count(p_, "t_stack") == 1 && !(:t_stack in p_.plan.dead_stacks) && ds_count(p_, "s_stack") < 3 && ds_same(p_.outs, b.outs), "partial init host: t_stack keeps its store (got $(ds_count(p_, "t_stack"))), s_stack is still reduced (got $(ds_count(p_, "s_stack")))")
    # the allocation of a stack with no access shrinks
    init = Meta.parse("function initstacks_ds_f(n, m)\n    s_stack = CuArray{Float64}(undef, (m * n + m,))\n    t_stack = CuArray{Float64}(undef, (m,))\n    return (s_stack, t_stack)\nend")
    sh = STADE.cgen_shrink_stacks(init, a.plan.dead_stacks)
    check(occursin("t_stack = CuArray{Float64}(undef, (1,))", string(sh)) && occursin("(m * n + m,)", string(sh)), "shrink: t_stack gets one slot, s_stack keeps its size")
    check(STADE.cgen_shrink_stacks(init, Symbol[]) == init, "shrink: with no dead stack the host is unchanged")
    for (label, src, n0, n1) in ds_cases()
        c1 = ds_run(src, args; dead = 1)
        c0 = ds_run(src, args; dead = 0)
        cc = ds_cpu(src, args)
        check(ds_count(c0, "s_stack") >= 3, "$label: the plan with dead_stack = 0 has $(ds_count(c0, "s_stack")) accesses")
        check(ds_count(c1, "s_stack") == ds_count(c0, "s_stack") && ds_count(c1, "t_stack") == 0, "$label: s_stack is left alone ($(ds_count(c1, "s_stack"))), t_stack is stripped")
        check(ds_same(c1.outs, c0.outs), "$label: the outputs are the same numbers with dead_stack = 0 and 1")
        check(isempty(c1.conflicts), "$label: the shim reports no conflict")
    end
    bad += ds_compact_checks()
    return bad
end

function validate_dead_stack(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += ds_unit_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("the keyword is not tested", "dead_scalars && dead_stack >= 1 &&", "dead_scalars &&"),
            ("an array no init host allocates may be a stack", "    filter!(x -> haskey(alloc, x) ||", "    filter!(x -> true ||"),
            ("a read need not repeat a store", "all(r -> r in get(stores, x, Set{Any}()), get(reads, x, Set{Any}()))", "true"),
            ("a store a read repeats may be deleted", "!(f in get(ctx.reads, x, Set{Any}()))", "true"),
            ("a local scalar may stand in an index formula", "            x in lvars || (ok[] = false)\n", ""),
            ("a prefix may differ between the accesses of one stack", "(length(ps) == 1 && !(:NOSPLIT in ps)) || continue", ""),
            ("the allocation prefix may differ from the formula prefix", "if cgen_dstack_inline_all(sub, idefs) == ph", "if true"),
            ("a host scalar assigned twice may stand in an index formula", "        isempty(rs) && return true\n        (length(rs) == 1 && depth < 8 && rs[1] !== :loop) || return false\n", "        return true\n"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases[intersect(eachindex(cases), slice)]
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_dead_stack.jl"))))
                    n = ds_unit_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all dead_stack checks passed" : "*** $bad dead_stack check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_dead_stack() == 0 ? 0 : 1)
end
