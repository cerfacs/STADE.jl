# ==================== validate_reuse_stored.jl =====================
# Checks the reuse of a stored value (`reuse_stored`, v0.4.38): a backward kernel that computes a forward sum again, `s = b[j]; for i ...; s = s + w * x[i]; end`, and
# uses `tanh(s)`, reads `h[j]` instead when a forward kernel launched earlier stored `h[j] = tanh(s)` from the same sum. The dead-scalar pass then deletes the sum.
#
#   julia validate_reuse_stored.jl           units, refusals, direct recognizer cases, sabotage
#   validate_reuse_stored(; phases = (:units,))
#
# Every case is a three-layer perceptron in the form the adjoint stage generates (`rs_base`), with one edit. It runs through the CPU shim (ownership_shim.jl) twice from the
# same random inputs, with `reuse_stored = 1` and with `reuse_stored = 0`, and once on the CPU as the source function.
#   - a case that must rewrite: the launch of the backward kernel of the layer has the stored array as its last argument, the kernel no longer holds the sum, and the
#     outputs of the two plans are the same numbers (the sum is not reordered), they agree with the CPU run to 1e-12, the shim reports no conflict, and outputs changed;
#   - a case that must refuse: the layer takes no extra argument. The layer of the other array (`h1`) still rewrites in every case, so a refusal is not a plan that
#     rewrote nothing at all. The outputs of the two plans are the same numbers.
# SABOTAGE edits of the recognizer, each of which one case must catch.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
isdefined(Main, :con_init!) || include(joinpath(@__DIR__, "validate_contraction.jl"))
using Random

# the adjoint of a perceptron with three layers (the form of the adjoint stage), n0 inputs, n1 and n2 hidden units, n3 outputs
const RS_BASE = """function rs_mlp(x, xb, w1, w1b, b1, b1b, w2, w2b, b2, b2b, w3, w3b, b3, b3b, h1, h1b, h2, h2b, y, yb, n0, n1, n2, n3)
    s = 0.0
    sb = 0.0
    for j = 1:n1
        s = b1[j]
        for i = 1:n0
            s = s + w1[(i - 1) * n1 + j] * x[i]
        end
        h1[j] = tanh(s)
    end
    for j = 1:n2
        s = b2[j]
        for i = 1:n1
            s = s + w2[(i - 1) * n2 + j] * h1[i]
        end
        h2[j] = tanh(s)
    end
    for j = 1:n3
        s = b3[j]
        for i = 1:n2
            s = s + w3[(i - 1) * n3 + j] * h2[i]
        end
        y[j] = s
        __oldb_0 = yb[j]
        yb[j] = 0.0
        sb = sb + __oldb_0
        for i = 1:n2
            __cse_0 = w3[(i - 1) * n3 + j]
            __cse_1 = h2[i]
            w3b[(i - 1) * n3 + j] = w3b[(i - 1) * n3 + j] + __cse_1 * sb
            h2b[i] = h2b[i] + __cse_0 * sb
        end
        __oldb_0 = sb
        sb = 0.0
        b3b[j] = b3b[j] + __oldb_0
    end
    for j = 1:n2
        s = b2[j]
        for i = 1:n1
            s = s + w2[(i - 1) * n2 + j] * h1[i]
        end
        __oldb_0 = h2b[j]
        h2b[j] = 0.0
        sb = sb + (1.0 - tanh(s) ^ 2) * __oldb_0
        for i = 1:n1
            __cse_2 = w2[(i - 1) * n2 + j]
            __cse_3 = h1[i]
            w2b[(i - 1) * n2 + j] = w2b[(i - 1) * n2 + j] + __cse_3 * sb
            h1b[i] = h1b[i] + __cse_2 * sb
        end
        __oldb_0 = sb
        sb = 0.0
        b2b[j] = b2b[j] + __oldb_0
    end
    for j = 1:n1
        s = b1[j]
        for i = 1:n0
            s = s + w1[(i - 1) * n1 + j] * x[i]
        end
        __oldb_0 = h1b[j]
        h1b[j] = 0.0
        sb = sb + (1.0 - tanh(s) ^ 2) * __oldb_0
        for i = 1:n0
            __cse_4 = w1[(i - 1) * n1 + j]
            __cse_5 = x[i]
            w1b[(i - 1) * n1 + j] = w1b[(i - 1) * n1 + j] + __cse_5 * sb
            xb[i] = xb[i] + __cse_4 * sb
        end
        __oldb_0 = sb
        sb = 0.0
        b1b[j] = b1b[j] + __oldb_0
    end
    return nothing
end"""

# the `k`-th occurrence of `old` in `src` becomes `new`; an occurrence that is not there is an error of the test
function rs_edit(src::String, old::String, new::String; nth::Int = 1)
    r = findall(old, src)
    length(r) >= nth || error("rs_edit: \"$old\" found $(length(r)) times, wanted occurrence $nth")
    return src[1:first(r[nth])-1] * new * src[last(r[nth])+1:end]
end

# random arguments: sizes n0 = 5, n1 = 7, n2 = 6, n3 = 4; gradients start from random values (the adjoint adds to them)
function rs_args(sd)
    r = MersenneTwister(sd)
    n0, n1, n2, n3 = 5, 7, 6, 4
    R(n) = randn(r, n)
    return Any[R(n0), R(8), R(n0 * n1), R(n0 * n1), R(n1 + 1), R(n1 + 1), R(n1 * n2), R(n1 * n2), R(n2 + 1), R(n2 + 1), R(n2 * n3), R(n2 * n3), R(n3 + 1), R(n3 + 1),
               R(n1 + 1), R(n1 + 1), R(n2 + 1), R(n2 + 1), R(n3 + 1), R(n3 + 1), n0, n1, n2, n3]
end

# builds the plan, evaluates it into Main and runs it on the shim; returns (outs, launched, conflicts, plan, host text, kernel text)
function rs_run(src::String, args; reuse::Int = 1, dead::Bool = true, run::Bool = true)
    con_init!()
    return Base.invokelatest(rs_run_body, src, args, reuse, dead, run)
end
function rs_run_body(src::String, args, reuse::Int, dead::Bool, run::Bool)
    Main.CUDA.MAXT[] = name -> 1024
    cuda = STADE.cgen_backend_cuda()
    plan = STADE.stade_gpu(Meta.parse(src), cuda; warp_atomics = false, shared_private = false, split_reduce = 0, contraction = 0, dead_scalars = dead, reuse_stored = reuse)
    htxt = string(plan.host)
    ktxt = Dict{String,String}(string(k.args[1].args[1]) => string(k) for k in plan.kernels if k.head === :function)
    run || return (outs = nothing, launched = String[], conflicts = Any[], plan = plan, host = htxt, kernels = ktxt)
    for k in plan.kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host)
    host = plan.host.args[1].args[1]
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS)
    Base.invokelatest(getfield(Main, host), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    return (outs = outs, launched = copy(Main.CUDA.LAUNCHED), conflicts = copy(Main.CUDA.VIOLATIONS), plan = plan, host = htxt, kernels = ktxt)
end

# the source function run on the CPU
function rs_cpu(src::String, args)
    Core.eval(Main, Meta.parse(src))
    cp = Any[a isa AbstractArray ? copy(a) : a for a in args]
    Base.invokelatest(getfield(Main, :rs_mlp), cp...)
    return cp
end

rs_relerr(a, b) = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(a, b) if x isa AbstractArray; init = 0.0)
rs_changed(outs, args) = sum(count(x .!= y) for (x, y) in zip(outs, args) if x isa AbstractArray; init = 0)

# does a launch of a backward kernel end with the stored array `h`?
rs_takes(host::String, h::String) = occursin(Regex("_\\d+!\\([^()]*, $h\\)"), host)

# (name, edit of the source, what must rewrite: the arrays whose layer reads the stored array, the arrays whose layer takes the plain kernel)
rs_cases() = Tuple[
    # a store at another index than the thread's own (the index is not a plain variable)
    ("the store is at another index (h2[j + 1])", rs_edit(RS_BASE, "h2[j] = tanh(s)", "h2[j + 1] = tanh(s)"), ["h1"], ["h2"]),
    # the stored function differs from the one the backward kernel uses
    ("the stored function is another one (tanh(s) + 1)", rs_edit(RS_BASE, "h2[j] = tanh(s)", "h2[j] = tanh(s) + 1.0"), ["h1"], ["h2"]),
    # the forward sum is written differently
    ("the forward sum has another text", rs_edit(RS_BASE, "s = s + w2[(i - 1) * n2 + j] * h1[i]", "s = s + h1[i] * w2[(i - 1) * n2 + j]"), ["h1"], ["h2"]),
    # the backward sum starts from another value
    ("the backward sum starts from another value", rs_edit(RS_BASE, "s = b2[j]", "s = b2[j] + 0.5"; nth = 2), ["h1"], ["h2"]),
    # a kernel between the two writes the stored array
    ("a kernel between writes the stored array", rs_edit(RS_BASE, "b3b[j] = b3b[j] + __oldb_0\n    end\n", "b3b[j] = b3b[j] + __oldb_0\n    end\n    for i = 1:n2\n        h2[i] = h2[i] * 1.0\n    end\n"), ["h1"], ["h2"]),
    # a kernel between writes an array the sum reads
    ("a kernel between writes an array the sum reads", rs_edit(RS_BASE, "b3b[j] = b3b[j] + __oldb_0\n    end\n", "b3b[j] = b3b[j] + __oldb_0\n    end\n    for i = 1:n1 * n2\n        w2[i] = w2[i] * 1.0\n    end\n"), ["h1"], ["h2"]),
    # the forward loop has fewer iterations than the backward loop: some j were not stored
    ("the forward loop stops one thread early", rs_edit(RS_BASE, "for j = 1:n2\n        s = b2[j]", "for j = 1:n2 - 1\n        s = b2[j]"), ["h1"], ["h2"]),
    # a scalar the sum reads is defined by other statements in the two kernels
    ("a scalar of the sum has another definition", rs_edit(rs_edit(RS_BASE, "s = b2[j]\n", "cj = 2.0 * j\n        s = b2[j] + cj\n"), "s = b2[j]\n", "cj = 3.0 * j\n        s = b2[j] + cj\n"), ["h1"], ["h2"]),
    # the stored function reads more than the sum
    ("the stored function reads another array (tanh(s + b2[j]))", rs_edit(rs_edit(RS_BASE, "h2[j] = tanh(s)", "h2[j] = tanh(s + b2[j])"), "(1.0 - tanh(s) ^ 2) * __oldb_0\n        for i = 1:n1\n            __cse_2", "(1.0 - tanh(s + b2[j]) ^ 2) * __oldb_0\n        for i = 1:n1\n            __cse_2"), ["h1"], ["h2"]),
    # the forward kernel changes the sum after the store, and uses the new value
    ("the forward kernel changes the sum after the store and uses it", rs_edit(RS_BASE, "h2[j] = tanh(s)", "h2[j] = tanh(s)\n        s = s + 1.0\n        t2 = s * 2.0\n        xb[j] = xb[j] + t2"), ["h1"], ["h2"]),
    ("the forward kernel has a second store that reads the sum", rs_edit(RS_BASE, "h2[j] = tanh(s)", "h2[j] = tanh(s)\n        s = s + 1.0\n        xb[j] = xb[j] + s"), ["h1"], ["h2"]),
    # two stores to the stored array in the forward kernel
    ("the forward kernel stores twice", rs_edit(RS_BASE, "h2[j] = tanh(s)", "h2[j] = tanh(s)\n        h2[j] = h2[j] * 1.0"), ["h1"], ["h2"]),
    # the forward kernel writes an array the sum reads
    ("the forward kernel writes an array the sum reads", rs_edit(RS_BASE, "h2[j] = tanh(s)", "h2[j] = tanh(s)\n        b2[j] = b2[j] + 0.0"), ["h1"], ["h2"]),
    # the backward kernel writes the stored array
    ("the backward kernel writes the stored array", rs_edit(RS_BASE, "b2b[j] = b2b[j] + __oldb_0", "b2b[j] = b2b[j] + __oldb_0\n        h2[j] = h2[j] * 1.0"), ["h1"], ["h2"]),
    # the backward kernel uses the sum itself, not only tanh of it
    ("the backward kernel uses the sum elsewhere", rs_edit(RS_BASE, "(1.0 - tanh(s) ^ 2) * __oldb_0", "(1.0 - tanh(s) ^ 2) * __oldb_0 + s * 0.0"), ["h1"], ["h2"]),
    # tanh of the sum is used after a later assignment to the sum
    ("tanh of the sum is used after the sum changed", rs_edit(RS_BASE, "h1b[i] = h1b[i] + __cse_2 * sb\n        end\n", "h1b[i] = h1b[i] + __cse_2 * sb\n            s = s + __cse_2\n        end\n        sb = sb + tanh(s) * 0.0\n"), ["h1"], ["h2"]),
    # a host scalar the loops read changes between the two launches
    ("a host scalar of the sum changes between the launches", rs_edit(RS_BASE, "b3b[j] = b3b[j] + __oldb_0\n    end\n", "b3b[j] = b3b[j] + __oldb_0\n    end\n    n1 = n1 - 1\n"), String[], ["h2", "h1"]),
]

function rs_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    args = rs_args(1)
    # --- the plain case: both layers rewrite
    a = rs_run(RS_BASE, args; reuse = 1)
    b = rs_run(RS_BASE, args; reuse = 0)
    cpu = rs_cpu(RS_BASE, args)
    check(rs_takes(a.host, "h2") && rs_takes(a.host, "h1"), "base: both backward kernels take the stored array (h2 and h1)")
    check(!rs_takes(b.host, "h2") && !rs_takes(b.host, "h1") && a.host != b.host, "base: with reuse_stored = 0 the launches are the plain ones")
    bk = [k for (k, v) in a.kernels if occursin(r"_4(_p)?!$", k)]
    bk0 = [k for (k, v) in b.kernels if occursin(r"_4(_p)?!$", k)]
    check(!isempty(bk) && all(k -> !occursin("s = b2[j]", a.kernels[k]) && occursin("h2[j]", a.kernels[k]), bk), "base: the backward kernel of layer 2 reads h2[j] and has no sum ($(join(sort(bk), ", ")))")
    check(!isempty(bk0) && all(k -> occursin("s = b2[j]", b.kernels[k]) && !occursin("h2[j]", b.kernels[k]), bk0), "base: with reuse_stored = 0 the kernel keeps the sum")
    check(rs_relerr(a.outs, b.outs) == 0.0, "base: the outputs of the two plans are the same numbers (error $(rs_relerr(a.outs, b.outs)))")
    check(rs_relerr(a.outs, cpu) <= 1e-12, "base: the outputs agree with the CPU run (error $(round(rs_relerr(a.outs, cpu), sigdigits = 3)))")
    check(isempty(a.conflicts) && isempty(b.conflicts), "base: the shim reports no conflict")
    check(rs_changed(a.outs, args) > 100, "base: the run changed $(rs_changed(a.outs, args)) elements")
    a2 = rs_run(RS_BASE, rs_args(2)); b2 = rs_run(RS_BASE, rs_args(2); reuse = 0)
    check(rs_relerr(a2.outs, b2.outs) == 0.0, "base: a second seed gives the same numbers in the two plans")
    # --- the gates
    g1 = rs_run(RS_BASE, args; reuse = 1, dead = false, run = false)
    g0 = rs_run(RS_BASE, args; reuse = 0, dead = false, run = false)
    check(g1.host == g0.host && g1.kernels == g0.kernels, "gate: without the dead-scalar pass the plan is the one of reuse_stored = 0")
    # --- an array the kernel already takes: no new argument, the same replacement
    src_in = rs_edit(RS_BASE, "(1.0 - tanh(s) ^ 2) * __oldb_0", "(1.0 - tanh(s) ^ 2) * __oldb_0 + h2[j] * 0.0")
    r1 = rs_run(src_in, args; reuse = 1); r0 = rs_run(src_in, args; reuse = 0)
    k4 = [k for (k, v) in r1.kernels if occursin(r"_4(_p)?!$", k)]
    check(!isempty(k4) && all(k -> !occursin("s = b2[j]", r1.kernels[k]), k4) && !rs_takes(r1.host, "h2") && rs_relerr(r1.outs, r0.outs) == 0.0, "an array the kernel already takes: the sum is gone and no argument is added")
    # --- a scalar of the sum with the same definition in both kernels rewrites
    src_same = rs_edit(rs_edit(RS_BASE, "s = b2[j]\n", "cj = 2.0 * j\n        s = b2[j] + cj\n"), "s = b2[j]\n", "cj = 2.0 * j\n        s = b2[j] + cj\n")
    s1 = rs_run(src_same, args; reuse = 1); s0 = rs_run(src_same, args; reuse = 0)
    check(rs_takes(s1.host, "h2") && rs_takes(s1.host, "h1") && rs_relerr(s1.outs, s0.outs) == 0.0, "a scalar of the sum with the same definition in both kernels: the layer rewrites")
    # --- refusals
    for (name, src, yes, no) in rs_cases()
        x = rs_run(src, args; reuse = 1); y = rs_run(src, args; reuse = 0)
        okyes = all(h -> rs_takes(x.host, h), yes)
        okno = all(h -> !rs_takes(x.host, h), no)
        # a layer that is refused keeps its sum: every kernel that reads and clears h2b[j] (the backward kernel of layer 2) still computes `s = b2[j]...`
        "h2" in no && (okno &= all(kv -> !occursin("__oldb_0 = h2b[j]", kv[2]) || occursin(r"s = b2\[j\]", kv[2]), x.kernels))
        check(okyes && okno && rs_relerr(x.outs, y.outs) == 0.0 && isempty(x.conflicts) && rs_changed(x.outs, args) > 0, "refusal, $name: layers $(join(no, ", ")) take the plain kernel" * (isempty(yes) ? "" : ", $(join(yes, ", ")) still rewrites") * " (same numbers)")
    end
    return bad
end

# direct recognizer cases: guards that the source cannot reach
function rs_direct_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    cuda = STADE.cgen_backend_cuda()
    expr = Meta.parse(RS_BASE)
    plan = STADE.cgen_emit(STADE.cgen_gather_stencil(STADE.cgen_ingest(expr), true), cuda; reduction_threshold = STADE.cgen_reduction_threshold_default())
    kernels = Expr[STADE.cgen_dead_scalars(k, true) for k in plan.kernels]
    host = plan.host
    lm = cuda.launch_macro
    kdefs = Dict{Symbol,Expr}(k.args[1].args[1] => k for k in kernels if k.head === :function)
    hs = Any[x for x in host.args[2].args]
    launches = [i for (i, st) in enumerate(hs) if st isa Expr && STADE.cgen_reuse_launch_info(st, kdefs, lm) !== nothing]
    infos = [STADE.cgen_reuse_launch_info(hs[i], kdefs, lm) for i in launches]
    ibase(n) = findfirst(inf -> endswith(string(inf.base), "_$(n)!"), infos)
    pb = launches[ibase(4)]; pf = launches[ibase(2)]
    info_b = infos[ibase(4)]; info_f = infos[ibase(2)]
    parts_b = STADE.cgen_kernel_parts(kdefs[info_b.base])
    params_b = Symbol[parts_b.sig.args[2:end]...]
    sub_b = Dict{Symbol,Any}(params_b[i] => info_b.call.args[1+i] for i in eachindex(params_b))
    blk = STADE.cgen_reuse_blocks(parts_b.rest)[1]
    defs_b = STADE.cgen_reuse_closure(parts_b.rest, blk.k, Set{Symbol}(r for r in blk.reads if !(r in params_b)), Set(params_b))
    bs = Any[STADE.cgen_reuse_subst(parts_b.rest[blk.k], sub_b), STADE.cgen_reuse_subst(parts_b.rest[blk.k+1], sub_b)]
    m = STADE.cgen_reuse_match(info_f, kdefs, blk, bs, defs_b, sub_b, parts_b)
    check(m !== nothing && m.hname === :h2 && m.idx === :j, "direct: the forward kernel of layer 2 matches the backward kernel (control)")
    # a thread index defined differently
    pbad = merge(parts_b, (head = :(__tid = (blockIdx()).x * (blockDim()).x + (threadIdx()).x),))
    check(STADE.cgen_reuse_match(info_f, kdefs, blk, bs, defs_b, sub_b, pbad) === nothing, "direct: another thread index is refused")
    # a coarse variant of a kernel
    pname = Symbol(replace(string(info_b.base), r"!$" => "_p!"))
    cname = Symbol(replace(string(info_b.base), r"!$" => "_c!"))
    kd2 = copy(kdefs); kd2[cname] = Expr(:function, Expr(:call, cname, kdefs[pname].args[1].args[2:end]...), kdefs[pname].args[2])
    stc = STADE.cgen_reuse_replace(hs[pb], pname, cname)
    check(STADE.cgen_reuse_launch_info(hs[pb], kdefs, lm) !== nothing && stc != hs[pb] && STADE.cgen_reuse_launch_info(stc, kd2, lm) === nothing, "direct: a launch of a coarse variant (_c!) is not described, the same launch with the variant of the plan is (control)")
    return bad
end

function validate_reuse_stored(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += rs_unit_checks()
        println("-- direct")
        bad += rs_direct_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("the keyword is not tested", "dead_scalars && reuse_stored >= 1 &&", "dead_scalars &&"),
            ("the dead-scalar pass is not required", "backend.kernel_tag == \"cuda\" && dead_scalars && reuse_stored >= 1 &&", "backend.kernel_tag == \"cuda\" && reuse_stored >= 1 &&"),
            ("a kernel between may write an array the sum reads", "isempty(intersect(inf.writes, host_reads)) || (ok = false; break)", "true"),
            ("a kernel between may write the stored array", "any(b -> cand.harg in b.inf.writes, between) && continue", "false && continue"),
            ("a host scalar of the sum may change between the launches", " && !(st.args[1] in host_scalars))) && continue", ")) && continue"),
            ("the thread guards may differ", "cgen_reuse_subst(parts_f.guard, sub_f) == cgen_reuse_subst(parts_b.guard, sub_b) || return nothing", "true"),
            ("the thread indexes may differ", "cgen_reuse_subst(parts_f.head, sub_f) == cgen_reuse_subst(parts_b.head, sub_b) || return nothing", "true"),
            ("the sums may differ", "Any[cgen_reuse_subst(rf[bf.k], sub_f), cgen_reuse_subst(rf[bf.k+1], sub_f)] == blockstmts_b || return nothing", "true"),
            ("the scalars of the sum may be defined differently", "        cgen_reuse_subst(defs_f[u], sub_f) == cgen_reuse_subst(defs_b[u], sub_b) || return nothing\n", ""),
            ("the stored index may be an expression and the function may read more than the sum", "(idx isa Symbol && cgen_split_vars(F) == Set([v]) && cgen_dead_pure(F, true)) || return nothing", "true"),
            ("the sum may be assigned after the store", "    any(x -> cgen_reuse_assigns(x, v), after) && return nothing\n", ""),
            # no sabotage case for `length(stores) == 1`: without it the case "a second store that reads the sum" is still refused, by a guard that I did not isolate
            ("the stored array may be written twice", "    count(x -> cgen_reuse_writes!(x, Set{Symbol}()) ∋ hname, rf) == 1 || return nothing\n", ""),
            ("the forward kernel may write an array the sum reads", "    isempty(intersect(setdiff(wr, Set([hname])), bf.arrs)) || return nothing\n", ""),
            ("the backward kernel may write the stored array", "        (hp in wr) && return nothing\n", ""),
            ("a kernel that keeps the sum is accepted", "        has_block && return nothing\n", ""),
            ("tanh of the sum may be used after the sum changed", "        cgen_reuse_count(kd.args[2], cand.F) == allowed || return nothing\n", ""),
            ("a coarse variant is described", "    any(n -> endswith(string(n), \"_c!\"), names) && return nothing\n", ""),
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
                    include($(repr(joinpath(@__DIR__, "validate_reuse_stored.jl"))))
                    n = rs_unit_checks() + rs_direct_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all reuse_stored checks passed" : "*** $bad reuse_stored check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_reuse_stored() == 0 ? 0 : 1)
end
