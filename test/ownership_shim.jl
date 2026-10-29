# ==================== ownership_shim.jl ============================
# A CPU stand-in for the few CUDA.jl entry points STADE's emitted kernels use,
# built to answer one question about each launch: which element did which
# thread touch, and how.
#
# STADE trusts that a plain (non-atomic) array write inside a kernel is disjoint
# across threads (see the comment above `cgen_race_write_arrays`). Nothing checked
# that trust on a real run. This shim runs every launch sequentially, one logical
# thread at a time, records every array access by thread id, and reports an
# element as UNSAFE when the outcome could depend on thread order:
#
#   multi_writer      two or more threads plain-write the element
#   write_vs_other    one thread plain-writes it, another reads or atomically adds
#   read_vs_atomic    one thread reads it, another atomically adds
#
# The check is about the access pattern, not about the order the shim happened
# to run in, so it sees a lost-update hazard that a real warp would hide.
# Accumulations through `CUDA.@atomic` by several threads are fine: addition
# commutes. Each launch is its own scope (a kernel boundary is a barrier).
#
# Only what the generator emits is implemented: `@cuda threads=.. blocks=.. f(..)`,
# `threadIdx/blockIdx/blockDim().x`, `CUDA.@atomic a[i..] += v`,
# `CUDA.@allowscalar`, `CUDA.allowscalar`, and `CuArray` as a constructor.
# Anything else fails loudly instead of being guessed at.

module CUDA

using Random

export @cuda, threadIdx, blockIdx, blockDim, CuDynamicSharedArray, sync_threads

mutable struct State
    tid::Int
    bx::Int
    tx::Int
    bdim::Int
    log::Dict{Tuple{Int,Int},NTuple{3,Set{Int}}}
    ty::Int
    phase::Int                       # barriers passed by the current block
    k::Int                           # index of the running thread inside its block
    shared::Vector{Float64}          # the block's dynamic shared memory
    chans::Any                       # (resume, paused) channels of the running block, nothing outside a block launch
    detail::Dict{Tuple{Int,Int},NTuple{3,Set{NTuple{3,Int}}}}   # the same log with (thread, block, phase) per access, for launches with barriers
    gx::Int                          # blockIdx().x, .y and .z of the running block (bx is the linear block number)
    gy::Int
    gz::Int
end
const ST = State(0, 0, 0, 0, Dict{Tuple{Int,Int},NTuple{3,Set{Int}}}(), 1, 0, 0, Float64[], nothing, Dict{Tuple{Int,Int},NTuple{3,Set{NTuple{3,Int}}}}(), 1, 1, 1)
const VIOLATIONS = NamedTuple[]
const NEXTID = Ref(0)
const LAUNCHES = Ref(0)
const LAUNCHED = String[]   # name of every kernel launched, in order
# Order in which one launch runs its logical threads: :ascending, :descending or :shuffled. A kernel whose
# result depends on it has a race; one that passes under all three is only known to be order-insensitive here.
const ORDER = Ref(:ascending)
const RNG = Ref(Random.Xoshiro(20260101))

struct TArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    id::Int
end
TArray(a::Array{T,N}) where {T,N} = TArray{T,N}(a, (NEXTID[] += 1))
TArray{T}(::UndefInitializer, dims::Integer...) where {T} = TArray(Array{T}(undef, dims...))
TArray{T}(::UndefInitializer, dims::Tuple) where {T} = TArray(Array{T}(undef, dims...))
TArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = TArray(Array{T,N}(undef, dims...))
TArray{T,N}(::UndefInitializer, dims::Tuple) where {T,N} = TArray(Array{T,N}(undef, dims...))

Base.size(a::TArray) = size(a.data)
Base.IndexStyle(::Type{<:TArray}) = IndexLinear()
Base.similar(a::TArray, ::Type{T}, dims::Dims) where {T} = TArray(Array{T}(undef, dims))
Base.Array(a::TArray) = copy(a.data)

function note!(a::TArray, i::Int, slot::Int)
    ST.tid == 0 && return nothing
    ent = get!(() -> (Set{Int}(), Set{Int}(), Set{Int}()), ST.log, (a.id, i))
    push!(ent[slot], ST.tid)
    ST.chans === nothing || push!(get!(() -> (Set{NTuple{3,Int}}(), Set{NTuple{3,Int}}(), Set{NTuple{3,Int}}()), ST.detail, (a.id, i))[slot], (ST.tid, ST.bx, ST.phase))
    return nothing
end

# slots: 1 = plain write, 2 = plain read, 3 = atomic add
function Base.getindex(a::TArray, i::Int)
    @boundscheck checkbounds(a.data, i)
    note!(a, i, 2)
    return @inbounds a.data[i]
end
function Base.setindex!(a::TArray, v, i::Int)
    @boundscheck checkbounds(a.data, i)
    note!(a, i, 1)
    @inbounds a.data[i] = v
    return a
end

function atomic_add!(a::TArray, v, idx::Integer...)
    li = LinearIndices(a.data)[idx...]
    note!(a, li, 3)
    a.data[li] += v
    return nothing
end

macro atomic(ex)
    (ex isa Expr && ex.head == :(+=) && ex.args[1] isa Expr && ex.args[1].head == :ref) ||
        error("CUDA shim: only `CUDA.@atomic a[i...] += v` is implemented, got: $(ex)")
    ref = ex.args[1]
    return Expr(:call, GlobalRef(@__MODULE__, :atomic_add!), esc(ref.args[1]), esc(ex.args[2]),
                map(esc, ref.args[2:end])...)
end

macro allowscalar(ex)
    return esc(ex)
end
allowscalar(::Bool) = nothing

threadIdx() = (x = ST.tx, y = ST.ty)
blockIdx() = (x = ST.gx, y = ST.gy, z = ST.gz)
blockDim() = (x = ST.bdim,)

# Shared memory and the block barrier exist only inside a launch with a two-dimensional block, where each thread of a block is a task and
# sync_threads hands control back to the scheduler. Two accesses conflict when the threads differ and either the blocks differ or no barrier
# separates them.
function CuDynamicSharedArray(::Type, n::Integer, off::Integer = 0)
    off + n <= length(ST.shared) || error("CUDA shim: shared array of $n elements does not fit the $(length(ST.shared)) the launch asked for")
    return ST.shared
end
function sync_threads()
    ST.chans === nothing && error("CUDA shim: sync_threads outside a launch with a two-dimensional block")
    k = ST.k
    put!(ST.chans[2][k], :sync)
    take!(ST.chans[1][k])
    return nothing
end


# `@cuda launch=false f(args)` compiles without launching: it returns a FakeKernel that `registers` can be asked about. The register count of a
# kernel comes from REGS, a function of the kernel's name, so a test can give the plain kernel and its coarsened twin any pair of counts.
struct FakeKernel
    name::String
end
const PROBES = String[]   # name of every kernel compiled with launch=false, in order
const REGS = Ref{Function}(name -> (endswith(name, "_c!") || endswith(name, "_c")) ? 48 : 40)
registers(k::FakeKernel) = REGS[](k.name)
const MAXT = Ref{Function}(name -> 1024)   # limit of threads per block of a compiled kernel, by name; a test can lower it for the split twin
maxthreads(k::FakeKernel) = MAXT[](k.name)

macro cuda(args...)
    call = args[end]
    (call isa Expr && call.head == :call) || error("CUDA shim: @cuda expects a call, got: $(call)")
    th = 1
    bl = 1
    sh = 0
    probe = false
    for o in args[1:end-1]
        (o isa Expr && o.head == :(=)) || error("CUDA shim: unsupported @cuda option: $(o)")
        o.args[1] == :threads ? (th = o.args[2]) :
        o.args[1] == :blocks  ? (bl = o.args[2]) :
        o.args[1] == :shmem   ? (sh = o.args[2]) :
        (o.args[1] == :launch && o.args[2] === false) ? (probe = true) :
        error("CUDA shim: unsupported @cuda option: $(o.args[1])")
    end
    # the arguments are evaluated, as the real macro does to learn their types: an undefined name fails here too
    probe && return Expr(:call, GlobalRef(@__MODULE__, :probe), esc(call.args[1]), map(esc, call.args[2:end])...)
    return Expr(:call, GlobalRef(@__MODULE__, :launch), esc(call.args[1]),
                Expr(:tuple, map(esc, call.args[2:end])...), esc(th), esc(bl), esc(sh))
end

function probe(f, args...)
    push!(PROBES, string(nameof(f)))
    return FakeKernel(string(nameof(f)))
end

function launch(f, args::Tuple, th::Tuple, bl::Union{Integer,Tuple}, sh::Integer)
    empty!(ST.log); empty!(ST.detail)
    LAUNCHES[] += 1
    push!(LAUNCHED, string(nameof(f)))
    nx, ny = th
    n = nx * ny
    nbx, nby, nbz = bl isa Tuple ? (bl..., ntuple(_ -> 1, 3 - length(bl))...) : (bl, 1, 1)
    blocks = collect(1:nbx * nby * nbz)
    ORDER[] == :descending && reverse!(blocks)
    ORDER[] == :shuffled && Random.shuffle!(RNG[], blocks)
    for b in blocks
        ST.shared = zeros(Float64, sh ÷ 8)
        resume = [Channel{Nothing}(1) for _ in 1:n]
        paused = [Channel{Symbol}(1) for _ in 1:n]
        errs = Any[]
        for k in 1:n
            @async begin
                take!(resume[k])
                try
                    f(args...)
                catch e
                    push!(errs, e)
                end
                put!(paused[k], :done)
            end
        end
        ST.chans = (resume, paused)
        alive = trues(n)
        ST.phase = 0
        while any(alive)
            order = collect(1:n)
            ORDER[] == :descending && reverse!(order)
            ORDER[] == :shuffled && Random.shuffle!(RNG[], order)
            for k in order
                alive[k] || continue
                ST.k = k
                ST.tx = (k - 1) % nx + 1
                ST.ty = (k - 1) ÷ nx + 1
                ST.tid = (b - 1) * n + k
                ST.bx = b
                ST.gx = (b - 1) % nbx + 1
                ST.gy = ((b - 1) ÷ nbx) % nby + 1
                ST.gz = (b - 1) ÷ (nbx * nby) + 1
                ST.bdim = nx
                put!(resume[k], nothing)
                take!(paused[k]) === :done && (alive[k] = false)
                isempty(errs) || (ST.chans = nothing; ST.tid = 0; throw(errs[1]))
            end
            ST.phase += 1
        end
        ST.chans = nothing
    end
    ST.tid = 0
    analyze_detail(f, args)
    return nothing
end

# the conflicts of a launch with barriers: the same three kinds as `analyze`, counting only accesses that no barrier orders
function analyze_detail(f, args)
    conflict(a, b) = a[1] != b[1] && (a[2] != b[2] || a[3] == b[3])
    seen = Dict{Tuple{Symbol,Symbol}, NamedTuple}()
    for ((aid, li), (W, R, A)) in ST.detail
        kind = nothing
        for w in W, o in W
            conflict(w, o) && (kind = :multi_writer; break)
            kind === nothing || break
        end
        if kind === nothing
            for w in W, o in Iterators.flatten((R, A))
                conflict(w, o) && (kind = :write_vs_other; break)
                kind === nothing || break
            end
        end
        if kind === nothing
            for r in R, o in A
                conflict(r, o) && (kind = :read_vs_atomic; break)
                kind === nothing || break
            end
        end
        kind === nothing && continue
        arr = argname_of(f, args, aid)
        key = (arr, kind)
        thr = sort!(unique(Int[x[1] for x in Iterators.flatten((W, R, A))]))
        if haskey(seen, key)
            seen[key] = merge(seen[key], (count = seen[key].count + 1,))
        else
            seen[key] = (kernel = nameof(f), array = arr, kind = kind, count = 1, example_index = li, threads = thr[1:min(4, length(thr))])
        end
    end
    for v in values(seen)
        push!(VIOLATIONS, v)
    end
    return nothing
end

launch(f, args::Tuple, th::Integer, bl::Integer, sh::Integer) = launch_1d(f, args, th, bl)

function launch_1d(f, args::Tuple, th::Integer, bl::Integer)
    empty!(ST.log)
    LAUNCHES[] += 1
    push!(LAUNCHED, string(nameof(f)))
    tids = collect(1:(th * bl))
    ORDER[] == :descending && reverse!(tids)
    ORDER[] == :shuffled && Random.shuffle!(RNG[], tids)
    for tid in tids
        ST.tid = tid
        ST.bx = (tid - 1) ÷ th + 1
        ST.gx = ST.bx
        ST.gy = 1
        ST.gz = 1
        ST.tx = (tid - 1) % th + 1
        ST.bdim = th
        f(args...)
    end
    ST.tid = 0
    analyze(f, args)
    return nothing
end

function argname_of(f, args, aid)
    names = Base.method_argnames(first(methods(f)))[2:end]
    for (k, x) in enumerate(args)
        x isa TArray && x.id == aid && k <= length(names) && return names[k]
    end
    return :unknown
end

function analyze(f, args)
    seen = Dict{Tuple{Symbol,Symbol}, NamedTuple}()
    for ((aid, li), (W, R, A)) in ST.log
        kind = length(W) >= 2 ? :multi_writer :
               (!isempty(W) && !isempty(setdiff(union(R, A), W))) ? :write_vs_other :
               (!isempty(R) && !isempty(A) && length(union(R, A)) > 1) ? :read_vs_atomic :
               nothing
        kind === nothing && continue
        arr = argname_of(f, args, aid)
        key = (arr, kind)
        if haskey(seen, key)
            seen[key] = merge(seen[key], (count = seen[key].count + 1,))
        else
            seen[key] = (kernel = nameof(f), array = arr, kind = kind, count = 1,
                         example_index = li, threads = sort!(collect(union(W, R, A)))[1:min(4, length(union(W, R, A)))])
        end
    end
    for v in values(seen)
        push!(VIOLATIONS, v)
    end
    return nothing
end

end # module CUDA
