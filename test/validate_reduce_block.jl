#=
    validate_reduce_block(dir="val-corpus"; only=nothing, sabotage=true)

Structural checks of the one-atomic-per-block reduction kernel (reduce_block, CUDA). The device result is measured on a GPU by the chain's
device tasks and by validate_gpu_truth; this file proves the text properties a device run cannot see.

  1. REPLACE: every large-trip array-element reduction that `reduce_block = false` writes as a sync-free mapreduce becomes one launch of a block
     kernel, and nothing else in the host changes shape (same number of `if` branches, same atomic kernels).
  2. COLLECTIVE: a block kernel reaches its `__stade_rb_sum` call on every thread. No `return` precedes the call, and the one atomic sits under a
     `threadIdx().x == 1` guard after it. A thread that left early would deadlock the barrier.
  3. HELPERS: the plan carries the shuffle helpers exactly when a kernel calls them.
  4. OTHER: `reduce_block = false` and `sync_free_reduce = false` leave no block kernel, AMDGPU and Metal output has none, and the launch caps the grid.
=#
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

function validate_reduce_block(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    cuda = STADE.cgen_backend_cuda()
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    has_call(e, f) = e isa Expr && (e.head == :call && !isempty(e.args) && e.args[1] === f || any(a -> has_call(a, f), e.args))
    is_view_cell(e) = e isa Expr && e.head == :call && length(e.args) == 3 && e.args[1] === :view
    count_sf(e) = e isa Expr ? (e.head == :(.=) && is_view_cell(e.args[1]) ? 1 : 0) + sum(count_sf(a) for a in e.args; init = 0) : 0
    is_block(k) = has_call(k, :__stade_rb_sum)
    has_return(e) = e isa Expr && (e.head == :return || any(has_return, e.args))
    count_ifs(e) = e isa Expr ? (e.head == :if ? 1 : 0) + sum(count_ifs(a) for a in e.args; init = 0) : 0
    # statements of the kernel body (inbounds = false, so no wrapper)
    stmts(k) = filter(x -> !(x isa LineNumberNode), k.args[2].args)
    # helpers are `@inline function ...` macrocalls, so the definition sits one level down
    fdef(h) = h isa Expr ? (h.head == :function ? h : (i = findfirst(a -> fdef(a) !== nothing, h.args); i === nothing ? nothing : fdef(h.args[i]))) : nothing
    fname(sig) = sig.head == :where ? fname(sig.args[1]) : sig.args[1]
    hnames_of(hs) = Set(string(fname(fdef(h).args[1])) for h in hs if fdef(h) !== nothing)
    total = 0; plans_with = 0
    for name in names
        pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp))
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = true) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            kw = (reduction_threshold = 0, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0)
            on = STADE.stade_gpu(e, cuda; kw...)
            off = STADE.stade_gpu(e, cuda; reduce_block = false, kw...)
            blocks = filter(is_block, on.kernels)
            n_off = count_sf(off.host)
            total += n_off; n_off > 0 && (plans_with += 1)
            check(length(blocks) == n_off, "$name [$mode]: $(length(blocks)) block kernel(s) replace $n_off sync-free reduction(s)")
            check(count_sf(on.host) == 0, "$name [$mode]: no sync-free mapreduce left beside a block kernel")
            check(length(on.kernels) - length(blocks) == length(off.kernels), "$name [$mode]: the other kernels are the same in number")
            check(count_ifs(on.host) == count_ifs(off.host), "$name [$mode]: host branches unchanged")
            for k in blocks
                ss = stmts(k)
                ci = findfirst(s -> has_call(s, :__stade_rb_sum), ss)
                check(ci !== nothing && !any(has_return, ss[1:ci]), "$name [$mode]: no return before the collective")
                check(ci !== nothing && ci < length(ss) - 1 && ss[ci+1] isa Expr && ss[ci+1].head == :if &&
                      has_call(ss[ci+1].args[1], :(==)) && !has_return(ss[ci+1]), "$name [$mode]: the atomic sits under the thread-1 guard")
                check(count(s -> s isa Expr && s.head == :macrocall, filter(x -> x isa Expr, ss)) <= 1, "$name [$mode]: a single atomic site")
            end
            hnames = hnames_of(on.helpers)
            check(isempty(blocks) == !("__stade_rb_sum" in hnames), "$name [$mode]: shuffle helpers present exactly when a block kernel is")
            check(isempty(filter(is_block, off.kernels)) && !("__stade_rb_sum" in hnames_of(off.helpers)),
                  "$name [$mode]: reduce_block = false has no block kernel")
            nsf = STADE.stade_gpu(e, cuda; sync_free_reduce = false, kw...)
            check(isempty(filter(is_block, nsf.kernels)), "$name [$mode]: sync_free_reduce = false has no block kernel")
            for (label, be) in (("amdgpu", STADE.cgen_backend_amdgpu()), ("metal", STADE.cgen_backend_metal()))
                pl = try STADE.stade_gpu(e, be; reduction_threshold = 0, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0) catch; continue end
                check(isempty(filter(is_block, pl.kernels)), "$name [$mode]: $label output has no block kernel")
            end
            src = STADE.io_expr_to_source(on.host)
            if !isempty(blocks)
                check(occursin("min(max(1, ", src) && occursin(string(STADE.cgen_rb_max_blocks()), src), "$name [$mode]: the launch caps the grid")
            end
        end
    end
    println("\n$total block reduction(s) in $plans_with plan(s)")
    check(total > 0, "at least one corpus kernel has an array-element reduction")
    if sabotage
        for (label, sab) in (("gate forced off", "cgen_reduce_block_on(backend) = false"),
                             ("helpers dropped", "cgen_rb_helper_exprs() = Expr[]"))
            script = """
                include($(repr(joinpath(@__DIR__, "validate_reduce_block.jl"))))
                Core.eval(STADE, Meta.parse($(repr(sab))))
                println("RESULT ", validate_reduce_block($(repr(dir)); only = ["dotprod"], sabotage = false))
            """
            o = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
            m = match(r"RESULT (\d+)", o)
            check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"$label\": dotprod fails a check")
        end
    end
    println("\n", bad == 0 ? "all block reduce checks passed" : "*** $bad block reduce check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_reduce_block(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
