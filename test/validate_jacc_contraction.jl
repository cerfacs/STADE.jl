include(joinpath(@__DIR__, "sabotage_run.jl"))
# ==================== validate_jacc_contraction.jl =============================
# Checks the JACC contraction twin (`jacc_contraction` of stade_jacc, v0.4.72): the tiled kernels of the CUDA path (a 64 x 64 tile per block of 16 x 16 threads, split sums for a
# long inner length) recast for JACC: thread and block indices from the global indices (i, j[, k]), the shared array from JACC.shared, the barrier JACC.sync_workgroup.
#
#   julia validate_jacc_contraction.jl             units, launch checks, sabotage
#   validate_jacc_contraction(; phases = (:units,))
#
# The recognizer and the emitter are the CUDA ones, so every case of validate_contraction.jl (forward products) and validate_adjoint_contraction.jl (adjoint groups) applies:
# the same kernels must tile and the same must be refused. Each case runs through the CPU stand-in for JACC of validate_jacc_split.jl (blocks of tasks, barriers, a record of
# who touched what; extended with three-dimensional launches) twice, from the same random inputs, with the twin forced on (`jacc_contraction = 1`) and off (0).
#   - a case that must tile: the same kernels exist and ran as in the CUDA test, the two runs agree to 1e-12 (the sum order differs), the stand-in reports no conflict between
#     threads that no barrier orders, NO thread returned early (JACC returns for an index outside the range, and a thread that skips a barrier hangs the block), and some
#     output changed;
#   - a case that must not tile: no twin exists and the two plans are the very same text.
# Launch checks: whole blocks of 16 x 16, the extents in the order (larger, smaller) with the block axes exchanged (and the kernel told) when the column tiles are fewer than the
# row tiles, the shared memory of the CUDA twin, a backend that is not CUDA, the limit on the shared memory, one launch site, and the twin together with the split twin.
# SABOTAGE edits of the JACC code, each of which one check must catch.

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
isdefined(Main, :jsp_init!) || include(joinpath(@__DIR__, "validate_jacc_split.jl"))
isdefined(Main, :CON_CASES) || include(joinpath(@__DIR__, "validate_contraction.jl"))
isdefined(Main, :ADJ_CASES) || include(joinpath(@__DIR__, "validate_adjoint_contraction.jl"))

jcn_run(args...; kw...) = (jsp_init!(); Base.invokelatest(jcn_run_body, args...; kw...))
function jcn_run_body(src::String, entry::Symbol, args, contraction::Int; split::Int = 0, backend = :cuda, mutate = nothing, warp::Bool = false, fill::Int = 0)
    J = Main.JACC
    J.MAXT[] = name -> 1024
    J.FAIL[] = nothing
    J.BACKEND[] = backend === :cuda ? J.CUDABackend() : J.OtherBackend()
    empty!(J.SPECS)
    plan = STADE.stade_jacc(Meta.parse(src); inbounds = false, split_reduce = split, split_fallback = 0, split_min_inner = 0, split_max_out = 10^9, warp_atomics = warp, jacc_contraction = contraction, jacc_tile_fill = fill)
    kernels = mutate === nothing ? plan.kernels : mutate(plan.kernels)
    for k in kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host)
    host = plan.host.args[1].args[1]
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS); empty!(Main.CUDA.EARLY)
    Base.invokelatest(getfield(Main, host), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    return (outs = outs, launched = copy(Main.CUDA.LAUNCHED), conflicts = copy(Main.CUDA.VIOLATIONS), early = copy(Main.CUDA.EARLY), specs = copy(J.SPECS), plan = plan)
end

jcn_relerr(a, b) = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(a.outs, b.outs) if x isa AbstractArray; init = 0.0)
jcn_changed(a, args) = any(x != y for (x, y) in zip(a.outs, args) if x isa AbstractArray)
jcn_kname(k) = string(k.args[1].args[1])
jcn_is_tile(n) = occursin(r"_t\d*!$", n)
jcn_tiles(plan) = count(k -> jcn_is_tile(jcn_kname(k)) && :__jsh in k.args[1].args, plan.kernels)
jcn_src(e) = STADE.io_expr_to_source(e)

function jcn_unit_checks(; uslice = 1:typemax(Int))
    bad = 0
    ucount = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    only = get(ENV, "JCN_ONLY", "")
    # forward products
    for (name, src, entry, mk, must) in CON_CASES
        occursin(only, name) || continue
        (ucount += 1) in uslice || continue
        try
            args = mk(1)
            a = jcn_run(src, entry, args, 1)
            b = jcn_run(src, entry, args, 0)
            twins = jcn_tiles(a.plan)
            ran = any(n -> endswith(n, "_t!"), a.launched)
            if must === :plain
                check(twins >= 1 && !ran && jcn_relerr(a, b) == 0.0 && jcn_changed(a, args), "forward, $name: a twin exists, the plain kernel runs, outputs equal (twins $twins)")
            elseif must
                err = jcn_relerr(a, b)
                check(twins >= 1 && ran && jcn_changed(a, args) && err <= 1e-12 && isempty(a.conflicts) && !any(jcn_is_tile, a.early), "forward, $name: twins $twins, ran $ran, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts)), early $(count(jcn_is_tile, a.early))")
            else
                same_text = jcn_src(a.plan.host) == jcn_src(b.plan.host) && twins == 0
                check(same_text && !ran, "forward, $name: refused, plan unchanged (twins $twins)")
            end
        catch e
            check(false, "forward, $name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    # adjoint groups
    for (name, src, entry, mk, must) in ADJ_CASES
        occursin(only, name) || continue
        (ucount += 1) in uslice || continue
        try
            args = mk(1)
            a = jcn_run(src, entry, args, 1)
            b = jcn_run(src, entry, args, 0)
            twins = count(k -> adj_is_twin(jcn_kname(k)), a.plan.kernels)
            firsts = count(k -> endswith(jcn_kname(k), "_a!"), a.plan.kernels)
            ran = count(adj_is_twin, a.launched)
            nstage = count(k -> adj_is_stage(jcn_kname(k)), a.plan.kernels)
            ranstage = count(adj_is_stage, a.launched)
            ranfirst = any(n -> endswith(n, "_a!"), a.launched)
            if must === :plain
                check(twins >= 1 && ran == 0 && !ranfirst && jcn_relerr(a, b) == 0.0 && jcn_changed(a, args), "adjoint, $name: twins exist, the plain kernel runs, outputs equal (twins $twins)")
            elseif must isa Integer && must > 0
                err = jcn_relerr(a, b)
                want_stage = name in ADJ_STAGE ? 1 : 0
                check(twins == must && firsts == 1 && ran == must && ranfirst && nstage == want_stage && ranstage == want_stage && jcn_changed(a, args) && err <= 1e-12 && isempty(a.conflicts) && !any(jcn_is_tile, a.early),
                      "adjoint, $name: twins $twins (want $must), stage $nstage (want $want_stage), ran $ran, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts)), early $(count(jcn_is_tile, a.early))")
            else
                same_text = jcn_src(a.plan.host) == jcn_src(b.plan.host) && twins == 0 && firsts == 0
                check(same_text && ran == 0, "adjoint, $name: refused, plan unchanged (twins $twins)")
            end
        catch e
            check(false, "adjoint, $name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    return bad
end

function jcn_launch_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    bname, bsrc, bentry, _, _ = CON_CASES[1]
    # the grid decides the axes: 2 x 2 tiles, more row tiles than column tiles (swapped), more column tiles than row tiles, and a single tile
    shapes = [("row tiles > column tiles", 300, 70), ("column tiles > row tiles", 67, 300), ("equal", 130, 130), ("one tile", 40, 40)]
    for (label, m, n) in shapes
        try
            k = 19
            mk(sd) = (r = Random.Xoshiro(sd); Any[STADE.val_randn(r, m * k), STADE.val_randn(r, k * n), STADE.val_randn(r, n), zeros(m * n), m, n, k])
            args = mk(1)
            a = jcn_run(bsrc, bentry, args, 1)
            b = jcn_run(bsrc, bentry, args, 0)
            sp = [s for s in a.specs if jcn_is_tile(s[1])]
            bxn, byn = cld(n, 64), cld(m, 64)
            want_blocks = (max(bxn, byn), min(bxn, byn))
            err = jcn_relerr(a, b)
            geometry = !isempty(sp) && all(s -> s[2] == (16, 16) && s[3] == want_blocks && s[5] == (16 * want_blocks[1], 16 * want_blocks[2]) && s[4] == 8 * 2080, sp)
            check(geometry && err <= 1e-12 && isempty(a.conflicts) && !any(jcn_is_tile, a.early) && jcn_changed(a, args), "$label ($m x $n): blocks $(want_blocks), extents (larger, smaller), shared $(8 * 2080), error $(round(err, sigdigits = 3)), early $(count(jcn_is_tile, a.early))")
        catch e
            check(false, "$label: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    # split sums and a batch launch in three dimensions
    try
        name, src, entry, mk, _ = first(c for c in ADJ_CASES if c[1] == "split inner sum, tall")
        args = mk(1)
        a = jcn_run(src, entry, args, 1); b = jcn_run(src, entry, args, 0)
        sp = [s for s in a.specs if jcn_is_tile(s[1])]
        three = filter(s -> length(s[3]) == 3, sp)
        check(!isempty(three) && all(s -> s[2] == (16, 16, 1) && s[5][1] == 16 * s[3][1] && s[5][2] == 16 * s[3][2] && s[5][3] == s[3][3], three) && any(s -> s[3][3] > 1, three) && jcn_relerr(a, b) <= 1e-12 && !any(jcn_is_tile, a.early),
              "$name: three-dimensional launch, threads 16 x 16 x 1, whole blocks, pieces $(maximum(s[3][3] for s in three; init = 0))")
    catch e
        check(false, "split sums: error " * first(split(sprint(showerror, e), "\n")))
    end
    try
        name, src, entry, mk, _ = first(c for c in CON_CASES if c[1] == "batched product")
        args = mk(1)
        a = jcn_run(src, entry, args, 1); b = jcn_run(src, entry, args, 0)
        sp = [s for s in a.specs if jcn_is_tile(s[1])]
        check(!isempty(sp) && all(s -> length(s[3]) == 3 && s[2] == (16, 16, 1) && s[5][3] == s[3][3], sp) && jcn_relerr(a, b) <= 1e-12 && !any(jcn_is_tile, a.early) && isempty(a.conflicts),
              "$name: batch on the third axis, blocks $(sp[1][3])")
    catch e
        check(false, "batched: error " * first(split(sprint(showerror, e), "\n")))
    end
    # an adjoint group with a staged forward product, and an adjoint group on a backend that is not CUDA
    try
        name, src, entry, mk, _ = first(c for c in ADJ_CASES if c[1] == "a recomputed forward value, a repeated decode and temporaries")
        args = mk(1)
        a = jcn_run(src, entry, args, 1); b = jcn_run(src, entry, args, 0); o = jcn_run(src, entry, args, 1; backend = :other)
        check(any(adj_is_stage, a.launched) && any(n -> endswith(n, "_a!"), a.launched) && jcn_relerr(a, b) <= 1e-12 && !any(jcn_is_tile, a.early) && isempty(a.conflicts) && jcn_changed(a, args), "$name: stage, first kernel and products run, results agree")
        check(!any(jcn_is_tile, o.launched) && !any(n -> endswith(n, "_a!"), o.launched) && jcn_relerr(o, b) == 0.0, "$name: a backend whose name has no CUDA runs the plain kernel")
    catch e
        check(false, "adjoint stage: error " * first(split(sprint(showerror, e), "\n")))
    end
    # warp-aggregated atomics: the plain kernels of the plan may use them, the tiled kernels never do
    try
        name, src, entry, mk, _ = ADJ_CASES[1]
        args = mk(1)
        w = jcn_run(src, entry, args, 1; warp = true)
        tk = [k for k in w.plan.kernels if jcn_is_tile(jcn_kname(k)) && :__jsh in k.args[1].args]
        check(!isempty(tk) && !any(occursin("__stade_wa_add!", jcn_src(k)) for k in tk) && all(occursin("Atomix.@atomic", jcn_src(k)) for k in tk), "$name: with warp atomics on, the tiled kernels keep their plain atomics")
        wb = jcn_run(src, entry, args, 0)
        check(jcn_relerr(w, wb) <= 1e-12 && !any(jcn_is_tile, w.early) && isempty(w.conflicts), "$name: with warp atomics on, results agree")
    catch e
        check(false, "warp atomics: error " * first(split(sprint(showerror, e), "\n")))
    end
    # a backend whose name has no CUDA runs the plain kernel
    args = CON_CASES[1][4](1)
    base = jcn_run(bsrc, bentry, args, 0)
    o = jcn_run(bsrc, bentry, args, 1; backend = :other)
    check(!any(jcn_is_tile, o.launched) && jcn_relerr(o, base) == 0.0, "a backend whose name has no CUDA runs the plain kernel")
    # switched off: the plan is the plain plan, with no backend test in the host
    off = jcn_run(bsrc, bentry, args, 0)
    check(!occursin("__stade_jcu", jcn_src(off.plan.host)) && jcn_tiles(off.plan) == 0, "jacc_contraction = 0: no twin, no backend test")
    # the twin together with the split twin: one backend test, both kinds of twin possible, the results agree. The second case has both kinds in one plan (tiles and a split twin).
    for (name, src, entry, mk, _) in (SPL_CASES[1], first(c for c in ADJ_CASES if c[1] == "two loops, the second stays"))
        try
            sargs = mk(1)
            both = jcn_run(src, entry, sargs, 1; split = 4)
            sbase = jcn_run(src, entry, sargs, 0; split = 0)
            nflag = length(findall("__stade_jcu = ", jcn_src(both.plan.host)))
            kinds = name == SPL_CASES[1][1] || (jcn_tiles(both.plan) > 0 && any(k -> endswith(jcn_kname(k), "_s!"), both.plan.kernels))
            check(nflag == 1 && kinds && jcn_relerr(both, sbase) <= 1e-12 && isempty(both.conflicts) && !any(jcn_is_tile, both.early), "with the split twin ($name): one backend test ($nflag), both kinds of twin $(kinds), results agree")
        catch e
            check(false, "with the split twin ($name): error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    # text: no CUDA-only form is left in a tile kernel; the barrier and the shared array are JACC's
    pl = jcn_run(bsrc, bentry, args, 1).plan
    tk = [k for k in pl.kernels if jcn_is_tile(jcn_kname(k))]
    ts = join(jcn_src.(tk), "\n")
    check(!isempty(tk) && !any(occursin(w, ts) for w in ("threadIdx", "blockIdx", "blockDim", "CuDynamicSharedArray", "sync_threads")) && occursin("JACC.shared(__jsh)", ts) && occursin("JACC.sync_workgroup()", ts),
          "tile kernel text: no CUDA thread or barrier form, JACC.shared and JACC.sync_workgroup")
    # a kernel launched from two sites is not tiled; a kernel with packed free variables is not tiled; the shared memory limit
    function dup_launch(e)
        e isa Expr || return e
        if e.head === :block
            out = Any[]
            for x in e.args
                push!(out, dup_launch(x))
                x isa Expr && x.head === :macrocall && STADE.jgen_split_launch_parts(x) !== nothing && push!(out, deepcopy(x))
            end
            return Expr(:block, out...)
        end
        return Expr(e.head, Any[dup_launch(x) for x in e.args]...)
    end
    plain = jcn_run(bsrc, bentry, args, 0).plan
    host2 = dup_launch(deepcopy(plain.host))
    nsites = length(findall("JACC.@parallel_for", jcn_src(host2)))
    r2 = STADE.jgen_contraction_apply(host2, Expr[deepcopy(k) for k in plain.kernels]; cmin = 1)
    check(nsites > length(findall("JACC.@parallel_for", jcn_src(plain.host))) && r2[1] == host2 && length(r2[2]) == length(plain.kernels), "a kernel launched from two sites: no twin ($nsites sites)")
    packed = Expr[]
    for k in plain.kernels
        k2 = deepcopy(k)
        k2.head === :function && k2.args[1] isa Expr && k2.args[1].head === :call && push!(k2.args[1].args, :__jargs)
        push!(packed, k2)
    end
    r3 = STADE.jgen_contraction_apply(deepcopy(plain.host), packed; cmin = 1)
    check(length(r3[2]) == length(packed), "a packed kernel (`__jargs` parameter): no twin")
    Core.eval(STADE, :(jgen_contraction_max_shmem() = 100))
    r4 = try Base.invokelatest(STADE.jgen_contraction_apply, deepcopy(plain.host), Expr[deepcopy(k) for k in plain.kernels]; cmin = 1) finally Core.eval(STADE, :(jgen_contraction_max_shmem() = 49152)) end
    check(length(r4[2]) == length(plain.kernels) && r4[1] == plain.host, "shared memory above the limit: no twin")
    # JACC launches f(i, j, k, x...) and a device compiles that splat only for at most 32 arguments: pad the plain kernels with unused arguments, so that the launches of the
    # twin sit below, at and above the limit. This is the shape of a doubled (HVP) seed kernel, which carried 33.
    function widen(plain, x)
        ks = Expr[deepcopy(k) for k in plain.kernels]
        for k in ks
            k.head === :function && k.args[1] isa Expr && k.args[1].head === :call && append!(k.args[1].args, [Symbol("__pad", i) for i in 1:x])
        end
        h = deepcopy(plain.host)
        function pad(e)
            e isa Expr || return
            pp = STADE.jgen_split_launch_parts(e)
            if pp !== nothing
                append!(pp.call.args, [Symbol("__pad", i) for i in 1:x])
                return
            end
            foreach(pad, e.args)
        end
        pad(h)
        return h, ks
    end
    function launched_args(h, names)
        mx = 0
        function walk(e)
            e isa Expr || return
            pp = STADE.jgen_split_launch_parts(e)
            if pp !== nothing
                pp.call.args[1] in names && (mx = max(mx, length(pp.call.args) - 1))
                return
            end
            # the launch of a tile kernel: JACC.parallel_for(spec, dims, kernel, arguments...)
            if e.head === :call && length(e.args) >= 4 && e.args[1] == Expr(:., :JACC, QuoteNode(:parallel_for)) && e.args[4] in names
                mx = max(mx, length(e.args) - 4)
                return
            end
            foreach(walk, e.args)
        end
        walk(h)
        return mx
    end
    for (label, src, entry, mk) in ((bname, bsrc, bentry, nothing), (let c = first(c for c in ADJ_CASES if c[1] == "split inner sum, tall"); ("adjoint: " * c[1], c[2], c[3], c[4]); end))
        try
            args = mk === nothing ? CON_CASES[1][4](1) : mk(1)
            plain = jcn_run(src, entry, args, 0).plan
            base_names = Set(k.args[1].args[1] for k in plain.kernels if k.head === :function)
            lim = STADE.jgen_contraction_max_args()
            probe(x) = begin
                h, ks = widen(plain, x)
                nh, nk = STADE.jgen_contraction_apply(h, ks; cmin = 1)
                added = Set(k.args[1].args[1] for k in nk if k.head === :function && !(k.args[1].args[1] in base_names))
                (twin = !isempty(added), most = launched_args(nh, added))
            end
            p0 = probe(0)
            x0 = lim - p0.most                       # every launched count grows by the padding, so the widest launch reaches the limit here
            ps = [(x, probe(x)) for x in (x0 - 1, x0, x0 + 1, x0 + 5)]
            check(p0.twin && x0 > 0 && ps[1][2].twin && ps[1][2].most < lim && ps[2][2].twin && ps[2][2].most == lim && !ps[3][2].twin && !ps[4][2].twin,
                  "$label: the twin is kept up to $(x0) padding arguments (a launch of exactly $lim) and refused beyond, no launch passes more than $lim")
        catch e
            check(false, "$label: padded arguments: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    # the recast refuses a form it does not know: a block index along an axis the launch does not have
    kd = :(function f(x)
        z = blockIdx().z
        return nothing
    end)
    check(STADE.jgen_contraction_recast(kd, false) === nothing && STADE.jgen_contraction_recast(kd, true) !== nothing, "recast: blockIdx().z needs the three-dimensional launch")
    kd2 = :(function f(x)
        z = blockDim().x
        return nothing
    end)
    check(STADE.jgen_contraction_recast(kd2, false) === nothing, "recast: a form of the CUDA kernel that is left over refuses the twin")
    return bad
end

function validate_jacc_contraction(; phases = (:units, :launch, :sabotage), slice = 1:typemax(Int), uslice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += jcn_unit_checks(; uslice)
    end
    if :launch in phases
        println("-- launch")
        bad += jcn_launch_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("the barrier is dropped", "return Expr(:call, Expr(:., :JACC, QuoteNode(:sync_workgroup)))", "return Expr(:call, :identity, 0)"),
            ("the shared array is not taken from the scratch", "return Expr(:call, Expr(:., :JACC, QuoteNode(:shared)), :__jsh)", "return Expr(:call, :zeros, :Float64, 2080)"),
            ("the thread along the first axis uses 8 threads", "thr && return Expr(:call, :+, Expr(:call, :mod, Expr(:call, :-, gi, 1), 16), 1)", "thr && return Expr(:call, :+, Expr(:call, :mod, Expr(:call, :-, gi, 1), 8), 1)"),
            ("the tile index ignores the exchanged axes", "return Expr(:if, Expr(:call, :(==), :__jsw, 0), tile(gi), tile(other))", "return tile(gi)"),
            ("the exchange is never asked for", "Expr(:(=), sw, Expr(:if, Expr(:call, :<, bx, by), 1, 0))", "Expr(:(=), sw, 0)"),
            ("the block grid is not exchanged", "blk = three ? Expr(:tuple, bx, by, bz) : Expr(:tuple, bp, bq)", "blk = three ? Expr(:tuple, bx, by, bz) : Expr(:tuple, bx, by)"),
            ("the extents are not in the order (larger, smaller)", "Expr(:tuple, Expr(:call, :*, 16, bp), Expr(:call, :*, 16, bq))\n", "Expr(:tuple, Expr(:call, :*, 16, bx), Expr(:call, :*, 16, by))\n"),
            ("the batch axis is the first block", "return thr ? 1 : :__jacc_k", "return thr ? 1 : 1"),
            ("the third extent is not the block count", "Expr(:tuple, Expr(:call, :*, 16, bx), Expr(:call, :*, 16, by), bz) :", "Expr(:tuple, Expr(:call, :*, 16, bx), Expr(:call, :*, 16, by), 1) :"),
            ("the shared memory is not sized", "Expr(:kw, :shmem_size, t.shmem)", "Expr(:kw, :shmem_size, 0)"),
            ("the shared array is not passed", "name, args..., sh, (three ? Any[] : Any[sw])...)", "name, args..., (three ? Any[] : Any[sw])...)"),
            ("the flag of the exchange is not passed", "name, args..., sh, (three ? Any[] : Any[sw])...)", "name, args..., sh)"),
            ("the backend test of a forward twin is dropped", "return Expr(:if, Expr(:&&, flag, t.cond), Expr(:block, jgen_contraction_launch(t, t.name, args, tag[])...), Expr(:block, e))", "return Expr(:if, t.cond, Expr(:block, jgen_contraction_launch(t, t.name, args, tag[])...), Expr(:block, e))"),
            ("the backend test of an adjoint group is dropped", "return Expr(:if, Expr(:&&, flag, cond), Expr(:block, body...), Expr(:block, e))", "return Expr(:if, cond, Expr(:block, body...), Expr(:block, e))"),
            ("the seeds are not stored (the first kernel is not launched)", "            push!(body, Expr(:macrocall, e.args[1], e.args[2], e.args[3], Expr(:call, A.a.ka_name, args..., kargs...)))\n", "\n"),
            ("a product is launched without its scratch array", "append!(body, jgen_contraction_launch(t, t.name, Any[args..., d.scratch...], tag[]))", "append!(body, jgen_contraction_launch(t, t.name, Any[args...], tag[]))"),
            ("the forward stage is launched without its scratch array", "append!(body, jgen_contraction_launch(A.fs, A.fs.name, Any[args..., fsc...], tag[]))", "append!(body, jgen_contraction_launch(A.fs, A.fs.name, Any[args...], tag[]))"),
            ("a kernel launched from two sites is tiled", "launches[lname] = haskey(launches, lname) ? nothing : p\n", "launches[lname] = p\n"),
            ("the limit on the shared memory is not checked", "            t.shmem <= jgen_contraction_max_shmem() || continue\n", "\n"),
            ("a leftover CUDA form does not refuse the twin", "        cgen_count_symbol(body, s) == 0 || return nothing\n", "\n"),
            ("blockIdx().z is accepted without the third axis", "elseif ax === :z && three", "elseif ax === :z"),
            ("the tiled kernels go through the warp pass", "warp_atomics && (kernels = Expr[jgen_contraction_tile_kernel(k) ? k : jgen_warp_atomics(k) for k in kernels])", "warp_atomics && (kernels = Expr[jgen_warp_atomics(k) for k in kernels])"),
            ("a forward launch above the argument limit is tiled", "            length(p.call.args) - 1 + 1 + (t.bz === nothing ? 1 : 0) <= jgen_contraction_max_args() || continue\n", "\n"),
            ("an adjoint group above the argument limit is tiled", "        nmax <= jgen_contraction_max_args() || continue\n", "\n"),
            ("the backend test is added twice", "    any(x -> x isa Expr && x.head === :(=) && x.args[1] === jgen_split_backend_flag(), blk) && return host\n", "\n"),
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
                    include($(repr(joinpath(@__DIR__, "validate_jacc_contraction.jl"))))
                    n = jcn_launch_checks()
                    println("RESULT ", n)
                """
                out = sab_read(script)
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all JACC contraction checks passed" : "*** $bad JACC contraction check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_jacc_contraction() == 0 ? 0 : 1)
end
