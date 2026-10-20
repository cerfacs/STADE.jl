include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_warp_atomics(dir="val-corpus"; only=nothing, sabotage=true)

Structural gate for the warp-aggregated atomics of the CUDA backend. No GPU needed; `validate_corpus_gpu.jl` checks the numbers on a device.

For every corpus kernel, adjoint and HVP (`keep_push_pop = false`, `fuse_ii_loops = true`), three groups of checks:

  1. EQUIVALENCE: replacing each `__stade_wa_add!(A, i, v)` call of the default CUDA output by `CUDA.@atomic A[i] += v` gives exactly the output of
     `warp_atomics = false`. The pass changes nothing but the atomic sites.
  2. HELPERS: the helper definitions are present exactly when a call is; `warp_atomics = false`, AMDGPU and Metal output have none.
  3. SABOTAGE: with the emission function swapped to put the index and the value in the wrong order, check 1 must fail on a kernel with an atomic.
"""
function validate_warp_atomics(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    cuda = STADE.cgen_backend_cuda()
    wa = STADE.cgen_warp_add_name()
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    function unwa(e)
        e isa Expr || return e
        args = Any[unwa(a) for a in e.args]
        if e.head == :call && length(args) == 4 && args[1] === wa
            return Expr(:macrocall, cuda.atomic_macro, nothing, Expr(:(+=), Expr(:ref, args[2], args[3]), args[4]))
        end
        return Expr(e.head, args...)
    end
    ncalls(e) = e isa Expr ? (e.head == :call && !isempty(e.args) && e.args[1] === wa ? 1 : 0) + sum(ncalls(a) for a in e.args; init = 0) : 0
    src(e) = STADE.io_expr_to_source(e)
    total = 0; kernels_with = 0
    for name in names
        pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp))
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = true) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            on = STADE.stade_gpu(e, cuda)
            off = STADE.stade_gpu(e, cuda; warp_atomics = false)
            n = sum(ncalls(k) for k in on.kernels; init = 0)
            total += n; n > 0 && (kernels_with += 1)
            same = length(on.kernels) == length(off.kernels) && all(src(unwa(a)) == src(b) for (a, b) in zip(on.kernels, off.kernels)) && src(on.host) == src(off.host)
            check(same, "$name [$mode]: $n warp call(s), output equals the plain atomics after substitution")
            check((n > 0) == !isempty(on.helpers), "$name [$mode]: helpers present iff a call is ($(length(on.helpers)) definition(s))")
            check(isempty(off.helpers) && !any(k -> STADE.cgen_expr_calls(k, wa), off.kernels), "$name [$mode]: warp_atomics = false has no helper and no call")
            for (label, be) in (("amdgpu", STADE.cgen_backend_amdgpu()), ("metal", STADE.cgen_backend_metal()))
                pl = try STADE.stade_gpu(e, be) catch; continue end
                check(isempty(pl.helpers) && !any(k -> STADE.cgen_expr_calls(k, wa), pl.kernels), "$name [$mode]: $label output has no warp call")
            end
        end
    end
    println("\n$total warp call(s) in $kernels_with kernel plan(s)")
    check(total > 0, "at least one corpus kernel emits a warp call")
    if sabotage
        sab = "function cgen_atomic_add(backend, target, other)\n" *
              "    if cgen_warp_atomics_on(backend) && target isa Expr && target.head == :ref && length(target.args) == 2 && target.args[1] isa Symbol\n" *
              "        return Expr(:call, cgen_warp_add_name(), target.args[1], other, target.args[2])\n    end\n" *
              "    return Expr(:macrocall, backend.atomic_macro, nothing, Expr(:(+=), target, other))\nend"
        script = """
            include($(repr(joinpath(@__DIR__, "validate_warp_atomics.jl"))))
            Core.eval(STADE, Meta.parse($(repr(sab))))
            println("RESULT ", validate_warp_atomics($(repr(dir)); only = ["idm_tile"], sabotage = false))
        """
        out = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
        m = match(r"RESULT (\d+)", out)
        check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"index and value swapped\": idm_tile fails the equivalence check")
    end
    println("\n", bad == 0 ? "all warp atomics checks passed" : "*** $bad warp atomics check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_warp_atomics(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
