# ==================== validate_jacc_split.jl =============================
# Checks the JACC split reduction (`split_reduce` of stade_jacc, v0.4.71): the shared-memory twin that gives each output L lanes, launched with a LaunchSpec.
#
#   julia validate_jacc_split.jl                   units, launch checks, sabotage
#   validate_jacc_split(; phases = (:units,))
#
# The twin is built by cgen_split_kernel (style = :jacc), so every case of validate_split.jl applies: the same kernels must split, the same must be refused. Each
# case runs through a CPU stand-in for JACC built on the CUDA shim of ownership_shim.jl (blocks of tasks, barriers, a record of who touched what):
#   JACC.launch_spec / JACC.parallel_for(spec, (M, N), f, args...)  one thread per (i, j) of the padded range, a block of spec.threads
#   JACC.@parallel_for range = N f(args...)                          the plain launch, one-dimensional
#   JACC.shared(x) / JACC.sync_workgroup()                           the block's dynamic shared memory and its barrier
# A case that must split: a twin exists, the two runs (split on, split_reduce = 0) agree to 1e-12, the shim reports no conflict between threads that no barrier
# orders, the twin ran, and NO thread of a twin launch returned early (JACC returns for an index outside the range, and a thread that skips a barrier hangs the block).
# A case that must not split: the two plans are the very same text.
# Launch checks: a backend whose name is not CUDA runs the plain kernel; a launch that raises the out-of-resources error takes the fallback twin, then the plain
# kernel; any other launch error propagates.
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
include(joinpath(@__DIR__, "validate_split.jl"))

const JSP_INIT = Ref(false)
function jsp_init!()
    JSP_INIT[] && return
    spl_init!()
    shim = quote
        module JACC
        import ..CUDA
        struct CUDABackend end
        struct OtherBackend end
        const BACKEND = Ref{Any}(CUDABackend())
        const MAXT = Ref{Function}(name -> 1024)       # threads per block a kernel of this name can take; above it the launch raises the out-of-resources error
        const OORMSG = Ref("CUDA error: too many resources requested for launch (code 701, ERROR_LAUNCH_OUT_OF_RESOURCES)")   # the other spelling: "Block register count exceeds device limit (143 regs/thread * 512 threads/block = 73216 > 65536 regs/block)"
        const FAIL = Ref{Any}(nothing)                  # an error message to raise from every twin launch
        const SPECS = Any[]
        default_backend() = BACKEND[]
        macro init_backend() nothing end
        struct LaunchSpec
            threads::Any
            blocks::Any
            shmem::Int
        end
        launch_spec(; threads = 0, blocks = 0, shmem_size = 0, kw...) = LaunchSpec(threads, blocks, shmem_size)
        array(x::AbstractArray) = CUDA.TArray(copy(x))
        zeros(T, n::Integer) = CUDA.TArray(Base.zeros(T, n))
        to_host(x) = copy(x.data)
        function parallel_for(spec::LaunchSpec, dims::Tuple, f, args...)
            th = spec.threads isa Tuple ? spec.threads : (spec.threads, 1)
            bl = spec.blocks isa Tuple ? spec.blocks : (spec.blocks, 1)
            length(dims) == 2 && dims[1] < dims[2] && error("JACC shim: a range with M < N swaps the block axes on a V100; the twin must keep M >= N")
            push!(SPECS, (string(nameof(f)), th, bl, spec.shmem, dims))
            FAIL[] === nothing || error(FAIL[])
            th[1] * th[2] > MAXT[](string(nameof(f))) && error(OORMSG[])
            CUDA.launch(f, args, th, bl, spec.shmem; jdims = dims)
        end
        function launch_plain(N, f, args...)
            th = 4
            CUDA.launch(f, args, (th, 1), (cld(N, th), 1), 0; jdims = (N,))
        end
        macro parallel_for(args...)
            r = args[1]; call = args[end]
            N = r.args[2]
            return esc(Expr(:call, GlobalRef(@__MODULE__, :launch_plain), N, call.args[1], call.args[2:end]...))
        end
        function shared(x)
            sh = CUDA.CuDynamicSharedArray(Float64, length(x))
            if CUDA.ST.ty == 1
                for k in CUDA.ST.tx:CUDA.ST.bdim:length(x)
                    sh[k] = x[k]
                end
            end
            CUDA.sync_threads()
            return sh
        end
        sync_workgroup() = CUDA.sync_threads()
        end
        module Atomix
        macro atomic(ex)
            return esc(Expr(:macrocall, Expr(:., :CUDA, QuoteNode(Symbol("@atomic"))), nothing, ex))
        end
        end
    end
    for ex in shim.args
        ex isa LineNumberNode || Core.eval(Main, ex)
    end
    JSP_INIT[] = true
end

jsp_run(args...; kw...) = (jsp_init!(); Base.invokelatest(jsp_run_body, args...; kw...))
function jsp_run_body(src::String, entry::Symbol, args, split::Int; fallback::Int = 0, coal::Bool = true, maxt = nothing, backend = :cuda, fail = nothing, min_inner::Int = 0, async::Bool = false, oormsg = nothing)
    J = Main.JACC
    J.MAXT[] = maxt === nothing ? (name -> 1024) : maxt
    J.FAIL[] = fail
    J.OORMSG[] = oormsg === nothing ? "CUDA error: too many resources requested for launch (code 701, ERROR_LAUNCH_OUT_OF_RESOURCES)" : oormsg
    J.BACKEND[] = backend === :cuda ? J.CUDABackend() : J.OtherBackend()
    empty!(J.SPECS)
    plan = STADE.stade_jacc(Meta.parse(src); inbounds = false, split_reduce = split, split_fallback = fallback, split_min_inner = min_inner, split_max_out = 10^9, split_coalesce = coal, warp_atomics = false, jacc_contraction = 0, jacc_async = async)
    for k in plan.kernels; Core.eval(Main, k); end
    Core.eval(Main, plan.host)
    host = plan.host.args[1].args[1]
    dev = Any[a isa AbstractArray ? Main.CUDA.TArray(copy(a)) : a for a in args]
    empty!(Main.CUDA.LAUNCHED); empty!(Main.CUDA.VIOLATIONS); empty!(Main.CUDA.EARLY)
    Base.invokelatest(getfield(Main, host), dev...)
    outs = Any[a isa Main.CUDA.TArray ? copy(a.data) : a for a in dev]
    return (outs = outs, launched = copy(Main.CUDA.LAUNCHED), conflicts = copy(Main.CUDA.VIOLATIONS), early = copy(Main.CUDA.EARLY), specs = copy(J.SPECS), plan = plan)
end

jsp_istwin(n) = occursin(r"_s\d*!$", n)
jsp_is_transposed(plan) = any(k -> jsp_istwin(string(k.args[1].args[1])) && occursin("__jx = __jacc_j", STADE.io_expr_to_source(k)), plan.kernels)

function jsp_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (name, src, entry, mk, must) in SPL_CASES
        try
            args = mk(1)
            a = jsp_run(src, entry, args, 4)
            b = jsp_run(src, entry, args, 0)
            twins = count(k -> jsp_istwin(string(k.args[1].args[1])), a.plan.kernels)
            ran = any(jsp_istwin, a.launched)
            early = count(jsp_istwin, a.early)
            if must
                err = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(a.outs, b.outs) if x isa AbstractArray; init = 0.0)
                changed = sum(count(x .!= y) for (x, y) in zip(a.outs, args) if x isa AbstractArray; init = 0)
                if name == "weight stride zero at run time"
                    a0 = jsp_run(src, entry, Any[args[1:end-1]..., 0], 4)
                    b0 = jsp_run(src, entry, Any[args[1:end-1]..., 0], 0)
                    same0 = all(x == y for (x, y) in zip(a0.outs, b0.outs) if x isa AbstractArray)
                    check(twins >= 1 && !any(jsp_istwin, a0.launched) && same0 && isempty(a0.conflicts), "$name: stride 0 at run time takes the plain kernel (twins $twins)")
                    check(ran && err <= 1e-12 && isempty(a.conflicts) && changed > 0 && early == 0, "$name: stride nonzero runs the twin, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts)), changed $changed, early $early")
                else
                    check(twins >= 1 && ran && err <= 1e-12 && isempty(a.conflicts) && changed > 0 && early == 0, "$name: twins $twins, ran $ran, error $(round(err, sigdigits = 3)), conflicts $(length(a.conflicts)), changed $changed, early $early")
                    a16 = jsp_run(src, entry, args, 16)
                    err16 = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(a16.outs, b.outs) if x isa AbstractArray; init = 0.0)
                    check(any(jsp_istwin, a16.launched) && err16 <= 1e-12 && isempty(a16.conflicts) && count(jsp_istwin, a16.early) == 0, "$name: 16 lanes, error $(round(err16, sigdigits = 3)), conflicts $(length(a16.conflicts))")
                    a1 = jsp_run(src, entry, args, 4; coal = false)
                    err1 = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(a1.outs, b.outs) if x isa AbstractArray; init = 0.0)
                    check(any(jsp_istwin, a1.launched) && err1 <= 1e-12 && isempty(a1.conflicts) && count(jsp_istwin, a1.early) == 0, "$name: plain layout, error $(round(err1, sigdigits = 3)), conflicts $(length(a1.conflicts))")
                end
            else
                same_text = STADE.io_expr_to_source(a.plan.host) == STADE.io_expr_to_source(b.plan.host) && twins == 0
                check(same_text && !ran, "$name: refused, plan unchanged (twins $twins)")
            end
        catch e
            check(false, "$name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    return bad
end

# the kernels whose lane loops are at unit stride put the lanes on threadIdx().x; the others keep the outputs there
function jsp_coal_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (name, src, entry, mk, want) in SPL_COAL_CASES
        try
            args = mk(1)
            plain = jsp_run(src, entry, args, 0)
            on = jsp_run(src, entry, args, 16)
            off = jsp_run(src, entry, args, 16; coal = false)
            err(r) = maximum(maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) for (x, y) in zip(r.outs, plain.outs) if x isa AbstractArray; init = 0.0)
            changed = sum(count(x .!= y) for (x, y) in zip(on.outs, args) if x isa AbstractArray; init = 0)
            tr_on = jsp_is_transposed(on.plan)
            tr_off = jsp_is_transposed(off.plan)
            check(tr_on == want && !tr_off, "$name: layout with the keyword on: transposed $tr_on (expected $want), off: $tr_off")
            check(any(jsp_istwin, on.launched) && err(on) <= 1e-12 && isempty(on.conflicts) && changed > 0 && count(jsp_istwin, on.early) == 0, "$name: the twin runs, error $(round(err(on), sigdigits = 3)), conflicts $(length(on.conflicts)), changed $changed")
            check(err(off) <= 1e-12 && isempty(off.conflicts) && count(jsp_istwin, off.early) == 0, "$name: keyword off, error $(round(err(off), sigdigits = 3)), conflicts $(length(off.conflicts))")
            on4 = jsp_run(src, entry, args, 4)
            check(err(on4) <= 1e-12 && isempty(on4.conflicts) && count(jsp_istwin, on4.early) == 0, "$name: 4 outputs per block, error $(round(err(on4), sigdigits = 3)), conflicts $(length(on4.conflicts))")
        catch e
            check(false, "$name: error " * first(split(sprint(showerror, e), "\n")))
        end
    end
    return bad
end

# the launch of a twin: spec, range, shared memory, and the host-side decisions
function jsp_launch_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    name, src, entry, mk, _ = SPL_CASES[1]
    args = mk(1)
    base = jsp_run(src, entry, args, 0)
    sameout(r) = all(x == y || (x isa AbstractArray && maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1.0)) <= 1e-12) for (x, y) in zip(r.outs, base.outs))
    # the geometry of the launch: a block of 32 x L, a padded range, shared memory of 8 bytes per slot
    r = jsp_run(src, entry, args, 4; coal = false)
    sp = [s for s in r.specs if jsp_istwin(s[1])]
    n = args[end-1]
    check(!isempty(sp) && all(s -> s[2] == (32, 4) && s[3] == (cld(n, 32), 1) && s[5] == (32 * cld(n, 32), 4) && s[4] == 8 * 32 * 4, sp), "plain layout: block 32 x 4, blocks $(cld(n, 32)), range $(32 * cld(n, 32)) x 4, shared $(8 * 32 * 4) bytes ($(sp))")
    # the transposed layout (the lanes on threadIdx().x, the outputs on threadIdx().y): L outputs per block, ceil(n / L) blocks
    cn, csrc, centry, cmk, _ = SPL_COAL_CASES[1]
    cargs = cmk(1)
    rt = jsp_run(csrc, centry, cargs, 4; coal = true)
    spt = [s for s in rt.specs if jsp_istwin(s[1])]
    cn_out = cargs[end-1]
    check(!isempty(spt) && all(s -> s[2] == (32, 4) && s[3] == (cld(cn_out, 4), 1) && s[5] == (32 * cld(cn_out, 4), 4), spt), "coalesced layout: block 32 x 4, blocks $(cld(cn_out, 4)), range $(32 * cld(cn_out, 4)) x 4 ($(spt))")
    # a backend that is not CUDA runs the plain kernel
    o = jsp_run(src, entry, args, 4; backend = :other)
    check(!any(jsp_istwin, o.launched) && sameout(o), "a backend whose name has no CUDA runs the plain kernel")
    # out of resources: the first twin fails, the fallback twin (8 lanes) runs
    f = jsp_run(src, entry, args, 16; fallback = 8, maxt = nm -> (endswith(nm, "_s!") ? 64 : 1024))
    check(any(nm -> endswith(nm, "_s8!"), f.launched) && !any(nm -> endswith(nm, "_s!"), f.launched) && sameout(f) && isempty(f.conflicts), "out of resources: the fallback twin runs ($(unique(f.launched)))")
    # the same refusal as CUDA.jl words it before the launch (measured: transformer_block and mpnn_gather at 16 lanes)
    f2 = jsp_run(src, entry, args, 16; fallback = 8, maxt = nm -> (endswith(nm, "_s!") ? 64 : 1024), oormsg = "Block register count exceeds device limit (143 regs/thread * 512 threads/block = 73216 > 65536 regs/block). Reduce per-thread register use")
    check(any(nm -> endswith(nm, "_s8!"), f2.launched) && !any(nm -> endswith(nm, "_s!"), f2.launched) && sameout(f2) && isempty(f2.conflicts), "out of resources, CUDA.jl's wording: the fallback twin runs ($(unique(f2.launched)))")
    # out of resources for both twins: the plain kernel
    p = jsp_run(src, entry, args, 16; fallback = 8, maxt = nm -> (jsp_istwin(nm) ? 32 : 1024))
    check(!any(jsp_istwin, p.launched) && sameout(p), "out of resources for both twins: the plain kernel runs")
    # out of resources with no fallback twin: the plain kernel
    q = jsp_run(src, entry, args, 16; fallback = 0, maxt = nm -> (jsp_istwin(nm) ? 32 : 1024))
    check(!any(jsp_istwin, q.launched) && sameout(q), "out of resources, no fallback twin: the plain kernel runs")
    # any other error propagates
    ok = try
        jsp_run(src, entry, args, 4; fail = "CUDA error: an illegal memory access was encountered (code 700, ERROR_ILLEGAL_ADDRESS)")
        false
    catch e
        occursin("illegal memory access", sprint(showerror, e))
    end
    check(ok, "another launch error is not swallowed")
    # a lane count above 32 is not split: M >= N could not hold
    big = jsp_run(src, entry, args, 33)
    check(!any(jsp_istwin, big.launched) && sameout(big), "more than 32 lanes: no twin")
    # asynchronous launches: `sync = false` on every launch of the host, the twins' LaunchSpec included; the results do not change
    as = jsp_run(src, entry, args, 4; async = true)
    ast = STADE.io_expr_to_source(as.plan.host)
    nl = length(findall("JACC.@parallel_for", ast)) + length(findall("JACC.launch_spec", ast))
    check(nl > 0 && length(findall("sync = false", ast)) == nl && any(jsp_istwin, as.launched) && sameout(as) && isempty(as.conflicts), "async: sync = false on all $(nl) launches (macro and spec), results unchanged")
    bs = STADE.io_expr_to_source(base.plan.host)
    check(!occursin("sync", bs) && !occursin("sync", STADE.io_expr_to_source(jsp_run(src, entry, args, 4).plan.host)), "async off: no sync keyword anywhere")
    # a kernel launched from two sites, and a kernel whose free variables are packed into `__jargs`, are not split (the twin's launch replaces ONE site; a packed
    # kernel has no parameter list the twin could mirror): jgen_split_apply returns host and kernels untouched
    pl = base.plan
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
    host2 = dup_launch(deepcopy(pl.host))
    nsites = length(findall("JACC.@parallel_for", STADE.io_expr_to_source(host2)))
    r2 = STADE.jgen_split_apply(host2, Expr[deepcopy(k) for k in pl.kernels], 4; min_inner = 0, max_out = 10^9)
    check(nsites > length(findall("JACC.@parallel_for", STADE.io_expr_to_source(pl.host))) && r2[1] == host2 && length(r2[2]) == length(pl.kernels), "a kernel launched from two sites: no twin ($(nsites) sites)")
    packed = Expr[]
    for k in pl.kernels
        k2 = deepcopy(k)
        k2.head === :function && k2.args[1] isa Expr && k2.args[1].head === :call && push!(k2.args[1].args, :__jargs)
        push!(packed, k2)
    end
    r3 = try STADE.jgen_split_apply(deepcopy(pl.host), packed, 4; min_inner = 0, max_out = 10^9) catch e; nothing end
    check(r3 !== nothing && length(r3[2]) == length(packed), "a packed kernel (`__jargs` parameter): no twin")
    # split_reduce = 0 leaves the plan exactly as the plain plan
    check(STADE.io_expr_to_source(base.plan.host) == STADE.io_expr_to_source(jsp_run(src, entry, args, 1).plan.host) && !any(jsp_istwin, base.launched), "split_reduce below 2: no twin, plain host")
    return bad
end

function validate_jacc_split(; phases = (:units, :launch, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += jsp_unit_checks()
    end
    if :coal in phases
        println("-- coalesced layout")
        bad += jsp_coal_checks()
    end
    if :launch in phases
        println("-- launch")
        bad += jsp_launch_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("the barrier of the JACC twin is dropped", "barrier = jacc ? Expr(:call, Expr(:., :JACC, QuoteNode(:sync_workgroup))) : Expr(:call, :sync_threads)", "barrier = jacc ? Expr(:call, :identity, 0) : Expr(:call, :sync_threads)"),
            ("the range of the launch is not padded to whole blocks", "dims = Expr(:tuple, Expr(:call, :*, 32, nblk), tw.lanes)", "dims = Expr(:tuple, tw.n_iter, tw.lanes)"),
            ("the block count ignores the outputs per block", "nblk = Expr(:call, Expr(:., :Base, QuoteNode(:cld)), tw.n_iter, tw.nout)", "nblk = Expr(:call, Expr(:., :Base, QuoteNode(:cld)), tw.n_iter, 32)"),
            ("the shared memory is not sized", "Expr(:kw, :shmem_size, 8 * 32 * tw.lanes * tw.nslots)", "Expr(:kw, :shmem_size, 0)"),
            ("the scratch array is not passed", "        push!(args, scr)\n", "        nothing\n"),
            ("the backend test is dropped", "return Expr(:if, Expr(:&&, jgen_split_backend_flag(), t.cond), jgen_split_try(first_launch, chain), e)", "return Expr(:if, t.cond, jgen_split_try(first_launch, chain), e)"),
            ("every launch error takes the fallback", "handler = Expr(:block, Expr(:(=), msg, Expr(:call, :sprint, :showerror, ev)), Expr(:||, oor, Expr(:call, :rethrow)), other)", "handler = Expr(:block, other)"),
            ("CUDA.jl's wording of the refusal is not recognised", "Expr(:||, Expr(:call, :occursin, \"register count exceeds\", msg), Expr(:call, :occursin, \"OUT_OF_RESOURCES\", msg))", "Expr(:call, :occursin, \"OUT_OF_RESOURCES\", msg)"),
            ("the driver's wording of the refusal is not recognised", "Expr(:||, Expr(:call, :occursin, \"register count exceeds\", msg), Expr(:call, :occursin, \"OUT_OF_RESOURCES\", msg))", "Expr(:call, :occursin, \"register count exceeds\", msg)"),
            ("the fallback twin is never launched", "chain = Expr(:if, t.cond, jgen_split_try(jgen_split_twin_launch(t2, p.call), e), e)", "chain = e"),
            ("the lane index of the JACC twin is the block row", "Expr(:(=), ly, ly_e),", "Expr(:(=), ly, 1),"),
            ("the thread index of the transposed twin ignores the block number", "Expr(:call, :*, Expr(:call, :div, jti, 32), nout), jx) : :__jacc_i)", "jx) : :__jacc_i)"),
            ("the shared array is not taken from the scratch", "jacc ? Expr(:call, Expr(:., :JACC, QuoteNode(:shared)), :__jscr)", "jacc ? Expr(:call, :zeros, :Float64, 32 * L * nslots)"),
            ("a kernel launched from two sites is split", "haskey(launches, p.call.args[1]) ? (launches[p.call.args[1]] = nothing) : (launches[p.call.args[1]] = p)", "launches[p.call.args[1]] = p"),
            ("the twin's launch spec stays synchronous in async mode", "Expr(:parameters, e.args[2].args..., Expr(:kw, :sync, false))", "Expr(:parameters, e.args[2].args...)"),
            ("the plain launch stays synchronous in async mode", "return Expr(:macrocall, e.args[1], e.args[2], e.args[3], Expr(:(=), :sync, false), e.args[4])", "return e"),
            ("a packed kernel is split", "    :__jargs in fargs && return nothing\n", "\n"),
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
                    include($(repr(joinpath(@__DIR__, "validate_jacc_split.jl"))))
                    n = jsp_unit_checks() + jsp_coal_checks() + jsp_launch_checks()
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all JACC split reduction checks passed" : "*** $bad JACC split reduction check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_jacc_split() == 0 ? 0 : 1)
end
