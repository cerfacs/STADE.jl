include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_sync_free_reduce(dir="val-corpus"; only=nothing, sabotage=true)

Structural gate for the device-side large-trip reduction branch of the CUDA backend. No GPU needed; `validate_corpus_gpu.jl` checks the numbers on a device.

For every corpus kernel, adjoint and HVP (`keep_push_pop = false`, `fuse_ii_loops = true`), with `reduction_threshold = 0` so every array-element reduction takes the large-trip branch:

  1. COUNT: the number of sync-free forms (`view(A, i:i) .= op.(view(A, i:i), mapreduce(..., dims = 1, ...))`) in the default CUDA host equals the number of
     `@allowscalar` mapreduce or sum forms in the `sync_free_reduce = false` host. The pass replaces each one and adds none.
  2. KERNELS: the device kernels are identical with the flag on and off.
  3. OTHER: `sync_free_reduce = false`, AMDGPU and Metal output has no sync-free form, and a CUDA output keeps no `@allowscalar` reduction on an array-element target.
  4. SABOTAGE: with the gate forced off, check 1 must fail on a kernel with such a reduction.
"""
function validate_sync_free_reduce(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    cuda = STADE.cgen_backend_cuda()
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    has_call(e, f) = e isa Expr && (e.head == :call && !isempty(e.args) && e.args[1] === f || any(a -> has_call(a, f), e.args))
    is_view_cell(e) = e isa Expr && e.head == :call && length(e.args) == 3 && e.args[1] === :view
    # sync-free form: a broadcast assignment into a one-element view
    count_sf(e) = e isa Expr ? (e.head == :(.=) && is_view_cell(e.args[1]) ? 1 : 0) + sum(count_sf(a) for a in e.args; init = 0) : 0
    # host form: an allowscalar block that assigns an array element from a mapreduce or a sum
    function count_host(e)
        e isa Expr || return 0
        n = 0
        if e.head == :macrocall && length(e.args) >= 3 && e.args[3] isa Expr && (has_call(e.args[3], :mapreduce) || has_call(e.args[3], :sum))
            n += 1
        end
        return n + sum(count_host(a) for a in e.args; init = 0)
    end
    src(e) = STADE.io_expr_to_source(e)
    total = 0; plans_with = 0
    for name in names
        pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp))
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = true) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            on = STADE.stade_gpu(e, cuda; reduction_threshold = 0, reduce_block = false, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0)
            off = STADE.stade_gpu(e, cuda; reduction_threshold = 0, sync_free_reduce = false, reduce_block = false, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0)
            n_on = count_sf(on.host); n_off = count_host(off.host)
            total += n_off; n_off > 0 && (plans_with += 1)
            check(n_on == n_off, "$name [$mode]: $n_on sync-free form(s) replace $n_off allowscalar reduction(s)")
            check(length(on.kernels) == length(off.kernels) && all(src(a) == src(b) for (a, b) in zip(on.kernels, off.kernels)), "$name [$mode]: kernels unchanged")
            check(count_sf(off.host) == 0, "$name [$mode]: sync_free_reduce = false has no sync-free form")
            for (label, be) in (("amdgpu", STADE.cgen_backend_amdgpu()), ("metal", STADE.cgen_backend_metal()))
                pl = try STADE.stade_gpu(e, be; reduction_threshold = 0, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0) catch; continue end
                check(count_sf(pl.host) == 0, "$name [$mode]: $label output has no sync-free form")
            end
        end
    end
    println("\n$total large-trip reduction(s) in $plans_with plan(s)")
    check(total > 0, "at least one corpus kernel has an array-element reduction")
    if sabotage
        sab = "cgen_sync_free_on(backend) = false"
        script = """
            include($(repr(joinpath(@__DIR__, "validate_sync_free_reduce.jl"))))
            Core.eval(STADE, Meta.parse($(repr(sab))))
            println("RESULT ", validate_sync_free_reduce($(repr(dir)); only = ["dotprod"], sabotage = false))
        """
        out = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
        m = match(r"RESULT (\d+)", out)
        check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"gate forced off\": dotprod fails the count check")
    end
    println("\n", bad == 0 ? "all sync-free reduce checks passed" : "*** $bad sync-free reduce check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_sync_free_reduce(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
