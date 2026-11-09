include(joinpath(@__DIR__, "sabotage_run.jl"))
# ==================== validate_dead_pops.jl =====================
# Checks cgen_dead_pops: the restores of a scalar that the reverse sweep overwrites before it reads, and the stores of a stack that nothing reads then.
#   - units: the adjoint function runs before and after the pass on random arrays and every non-stack array must agree exactly. Kernels: a scalar reassigned
#     in an independent loop (the pops are dead and the forward loop must leave the host), a conditional reassignment, a running state that the next
#     iteration reads (the pops are live and stay), a scalar reassigned in an inner loop, an array of the caller whose name ends in `_stack` (never cleaned), a
#     stack that a read touches through an index the walk cannot match (left as it is).
#   - SABOTAGE: edits of the new code, each of which one check must catch.
#   julia validate_dead_pops.jl        units, sabotage
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# (label, source, length of every array as a function of n, remaining stores of a stack: :none or :some, forward loop leaves the host)
const DP_UNIT = [
    ("a scalar reassigned in an independent loop", n -> n, :none, true, """
function f(u, un, n)
    for i = 1:n
        s = u[i]
        s = s + 0.1 * s * s
        s = s * 0.5 + u[i]
        un[i] = s
    end
    return nothing
end"""),
    ("a chain of three reassignments", n -> n, :some, true, """
function f(u, w, un, n)
    for i = 1:n
        s = u[i] * w[i]
        s = s + 0.1 * s * s
        s = s * s + w[i]
        s = 0.3 * s + u[i]
        un[i] = s
    end
    return nothing
end"""),
    ("a conditional reassignment", n -> n, :any, false, """
function f(u, un, n)
    for i = 1:n
        t = u[i]
        if t > 0.0
            t = t * t + 1.0
        end
        un[i] = t * u[i]
    end
    return nothing
end"""),
    ("a running state that the next iteration reads", n -> n, :some, false, """
function f(u, un, n)
    s = 0.0
    for i = 1:n
        s = 0.9 * s + u[i]
        un[i] = s * s
    end
    return nothing
end"""),
    ("a scalar reassigned in an inner loop", n -> n, :some, false, """
function f(u, un, n)
    for i = 1:n
        s = u[i]
        for k = 1:3
            s = s + 0.1 * s * s
        end
        un[i] = s
    end
    return nothing
end"""),
    ("an array of the caller that ends in _stack", n -> n, :any, true, """
function f(u, tmp_stack, un, n)
    for i = 1:n
        s = u[i]
        s = s + 0.1 * s * s
        tmp_stack[i] = s
        s = s * 0.5 + u[i]
        un[i] = s
    end
    return nothing
end"""),
]

function dp_adjoint(src::String)
    adj = STADE.stade_adjoint(Meta.parse(src); keep_push_pop = false, fuse_ii_loops = true)
    ginit = STADE.stade_gpu(adj.initstacks, STADE.cgen_backend_cuda())
    ih = Dict{Symbol,Expr}(ginit.host.args[1].args[1] => ginit.host)
    return adj, ih
end

# the arrays that the initstacks function allocates
function dp_stack_names(init::Expr)
    out = Symbol[]
    for st in init.args[2].args
        st isa Expr && st.head === :(=) && st.args[1] isa Symbol && st.args[2] isa Expr && st.args[2].head === :call && :undef in st.args[2].args && push!(out, st.args[1])
    end
    return out
end

# run an adjoint function on random arrays; returns every non-stack array
function dp_run(fexpr::Expr, init::Expr, n::Int, len::Function, seed)
    m1, m2 = Module(), Module()
    f = Core.eval(m1, fexpr)
    fi = Core.eval(m2, init)
    names = dp_stack_names(init)
    args = fexpr.args[1].args[2:end]
    rng = MersenneTwister(seed)
    stk = Base.invokelatest(fi, n)
    stk = stk isa Tuple ? collect(stk) : [stk]
    vals = Any[]
    kept = Pair{Symbol,Any}[]
    si = 0
    for a in args
        if a === :n
            push!(vals, n)
        elseif a in names
            si += 1
            push!(vals, stk[findfirst(==(a), names)])
        else
            arr = randn(rng, len(n))
            push!(vals, arr)
            push!(kept, a => arr)
        end
    end
    Base.invokelatest(f, vals...)
    GC.@preserve m1 m2 nothing  # the modules must outlive both calls: a function of a collected module crashes the runtime
    return kept
end

# stores and reads of the stacks in a function: (stores, reads)
function dp_count(fexpr::Expr, stacks)
    stores = 0
    reads = 0
    function walk(e)
        e isa Expr || return
        if e.head === :(=) && e.args[1] isa Expr && e.args[1].head === :ref && e.args[1].args[1] in stacks
            stores += 1
            walk(e.args[2])
            return
        end
        e.head === :ref && e.args[1] in stacks && (reads += 1)
        for a in e.args
            walk(a)
        end
    end
    walk(fexpr)
    return stores, reads
end

function dp_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, mklen, want, leaves, src) in DP_UNIT
        try
            len = mklen
            adj, ih = dp_adjoint(src)
            init = adj.initstacks
            stacks = dp_stack_names(init)
            after = STADE.cgen_dead_pops(adj.adjoint, ih)
            worst = 0.0
            for n in (1, 2, 3, 7)
                a = dp_run(adj.adjoint, init, n, len, n)
                b = dp_run(after, init, n, len, n)
                for ((na, xa), (nb, xb)) in zip(a, b)
                    worst = max(worst, maximum(abs.(xa .- xb)))
                end
            end
            stores, reads = dp_count(after, Set(stacks))
            ok_stores = want === :none ? (stores == 0 && reads == 0) : want === :some ? (stores > 0 && reads > 0) : true
            check(worst == 0.0 && ok_stores, "$(label): max difference $(worst), $(stores) stack store(s) and $(reads) read(s) left (want $(want))")
            plan = STADE.stade_gpu(adj.adjoint, STADE.cgen_backend_cuda(); init_hosts = ih)
            host = occursin("allowscalar", STADE.io_expr_to_source(plan.host))
            leaves && check(!host, "$(label): the forward loop is a device kernel")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    # a read through an index that no store has the formula of: the stack must be left as it is
    try
        init = Meta.parse("function initstacks_g(n)\n    s_stack = Vector{Float64}(undef, n + 1)\n    return s_stack\nend")
        g = Meta.parse("""
function g(u, un, s_stack, n)
    for i = 1:n
        s_stack[i] = u[i]
    end
    k = 0
    for i = 1:n
        k = k + 1
        un[i] = s_stack[k]
    end
    return nothing
end""")
        after = STADE.cgen_dead_pops(g, Dict{Symbol,Expr}(:initstacks_g => init))
        worst = 0.0
        for n in (1, 2, 5)
            a = dp_run(g, init, n, m -> m + 1, n)
            b = dp_run(after, init, n, m -> m + 1, n)
            for ((na, xa), (nb, xb)) in zip(a, b)
                worst = max(worst, maximum(abs.(xa .- xb)))
            end
        end
        stores, reads = dp_count(after, Set([:s_stack]))
        check(worst == 0.0 && stores == 1 && reads == 1, "a read that no store matches: max difference $(worst), $(stores) store(s) and $(reads) read(s) left (want 1 and 1)")
    catch e
        check(false, "a read that no store matches: error $(first(split(sprint(showerror, e), "\n")))")
    end
    # the switch: with STADE_DEAD_POPS=0 the first kernel keeps its host loop
    old = get(ENV, "STADE_DEAD_POPS", nothing)
    ENV["STADE_DEAD_POPS"] = "0"
    try
        adj, ih = dp_adjoint(DP_UNIT[1][5])
        plan = STADE.stade_gpu(adj.adjoint, STADE.cgen_backend_cuda(); init_hosts = ih)
        check(occursin("allowscalar", STADE.io_expr_to_source(plan.host)), "STADE_DEAD_POPS=0 keeps the host loop")
    finally
        old === nothing ? delete!(ENV, "STADE_DEAD_POPS") : (ENV["STADE_DEAD_POPS"] = old)
    end
    return bad
end

function validate_dead_pops(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += dp_unit_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("a restore is dropped although the scalar is read", "if !(v in live) && cgen_dead_pure(s.args[2], refs_ok) && (droppable === nothing || droppable(v, s.args[2]))", "if cgen_dead_pure(s.args[2], refs_ok) && (droppable === nothing || droppable(v, s.args[2]))"),
            ("the stores of a stack that is read are dropped", "st[:mode] === :drop && lhs.args[1] in st[:eligible] && !(key in st[:reads]) && return nothing", "st[:mode] === :drop && lhs.args[1] in st[:eligible] && return nothing"),
            ("a caller's array that ends in _stack is cleaned", "return Set{Symbol}(x for x in params if x in alloc || (", "return Set{Symbol}(x for x in params if endswith(string(x), \"_stack\") || x in alloc || ("),
            ("the pass is off", "cgen_dead_pops_enabled() = get(ENV, \"STADE_DEAD_POPS\", \"1\") != \"0\"", "cgen_dead_pops_enabled() = false"),
            ("the pass is not applied before the offload decision", "    dead_scalars && (expr = cgen_dead_pops(expr, init_hosts))\n", ""),
            ("the index scalars are not expanded", "            env[lhs] = cgen_dp_canon(rhs, env, st)\n", "            nothing\n"),
            ("a read that no store matches is ignored", "all(r -> r in st[:stores], (r for r in st[:reads] if r[1] === k)) && push!(st[:eligible], k)", "push!(st[:eligible], k)"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases[intersect(eachindex(cases), slice)]
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && (check(true, "anchor \"$label\": found once"); continue)
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_dead_pops.jl"))))
                    n = dp_unit_checks()
                    println("RESULT ", n)
                """
                out = sab_read(script)
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all dead pop checks passed" : "*** $bad dead pop check(s) failed ***")
    return bad
end
abspath(PROGRAM_FILE) == abspath(joinpath(@__DIR__, "validate_dead_pops.jl")) && exit(validate_dead_pops(; phases = (:units, :sabotage)) == 0 ? 0 : 1)
