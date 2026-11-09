# Data-parallel gradient of a benchmark kernel through stade_ddp, against the plain stade_adjoint loop.
#
#   [STADE_KIT_KERNELS=DIR] julia validate_ddp_bench.jl [KERNEL.jl]   one process: the ranks run one after another (correctness only, no timing claim)
#   mpirun -n P julia validate_ddp_bench.jl [KERNEL.jl]      P ranks, MPI.Allreduce! (correctness and timing)
#
# The kernel is mlp3 of the benchmark kit (one sample per call). Sample s goes to rank mod(s-1, P). Each rank calls the unmodified adjoint on its
# samples; the :accum buffers (the gradients of the weights and biases) accumulate across its samples and are summed over ranks.
# Reference: one rank, all samples, the same adjoint. Both must agree to rtol.
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# MPI is used only when the process was started by an MPI launcher (or STADE_BENCH_MPI=1). MPI.Init in a plain process can crash or hang
# on a cluster node, so a single process falls back to two serial ranks.
const HAVE_MPI = (haskey(ENV, "OMPI_COMM_WORLD_SIZE") || haskey(ENV, "PMI_SIZE") || get(ENV, "STADE_BENCH_MPI", "") == "1") && try
    @eval using MPI
    true
catch
    false
end

function bb_bank(kernel, ints, values, seed)
    bank = Dict{Symbol,Any}()
    for a in kernel.sig.args
        k = kernel.sig.kinds[a]
        if k == :scalar_int
            bank[a] = ints[a]
        elseif k in (:scalar_float, :array_float)
            bank[a] = deepcopy(values[a]); bank[STADE.agen_shadow(a)] = deepcopy(seed[a])
        else
            bank[a] = deepcopy(values[a])
        end
    end
    return bank
end

# The benchmark kit keeps its kernels outside this tree; STADE_KIT_KERNELS names the directory (the variable validate_own_kit.jl reads).
bb_default_kernel() = joinpath(get(ENV, "STADE_KIT_KERNELS", joinpath(@__DIR__, "..", "..", "kernels")), "nn", "mlp3.jl")

function bench_batch(path::String = bb_default_kernel(); nsamples::Int = 32, dims = (512, 1024, 1024, 256), rtol::Float64 = 1e-12, reps::Int = 3)
    comm = nothing; rank = 0; P = 1
    if HAVE_MPI
        MPI.Initialized() || MPI.Init()
        comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm); P = MPI.Comm_size(comm)
    end
    serial_ranks = (comm === nothing || P == 1) ? 2 : 1     # one process: emulate two ranks one after another
    isfile(path) || error("benchmark kernel not found: $path (set STADE_KIT_KERNELS to the kit's kernels directory, or pass the file)")
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal); sig = kernel.sig
    out = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
    stack_names = Symbol[a for a in out.initstacks.args[1].args[2:end]]
    adjoint_fn = STADE.val_compile(out.adjoint); initstacks_fn = STADE.val_compile(out.initstacks)
    ep = STADE.stade_ddp(primal; per_sample = [:x], mode = :adjoint)
    reset_accum! = STADE.val_compile(ep.reset_accum.expr); reset_reduced! = STADE.val_compile(ep.reset_reduced.expr); reset_local! = STADE.val_compile(ep.reset_local.expr)
    roles = ep.roles
    n0, n1, n2, n3 = dims
    ints = Dict{Symbol,Any}(:n0 => n0, :n1 => n1, :n2 => n2, :n3 => n3)
    len = Dict(:x => n0, :w1 => n0 * n1, :b1 => n1, :w2 => n1 * n2, :b2 => n2, :w3 => n2 * n3, :b3 => n3, :h1 => n1, :h2 => n2, :y => n3)
    Random.seed!(7)                                          # the same draw on every rank
    values = Dict{Symbol,Any}(a => (a in (:w1, :w2, :w3) ? 0.05 .* STADE.val_randn(len[a]) : STADE.val_randn(len[a])) for a in keys(len))
    xs = [STADE.val_randn(n0) for _ in 1:nsamples]
    seed = STADE.val_zeros_like(kernel, values)
    for d in sig.dependents
        sig.kinds[d] == :array_float && (seed[d] = ones(length(values[d])))
    end
    accum = STADE.ddpgen_role_args(kernel, roles, :adjoint, :accum); reduced = STADE.ddpgen_role_args(kernel, roles, :adjoint, :reduced)
    local_bufs = STADE.ddpgen_role_args(kernel, roles, :adjoint, :local)

    # this rank's samples -> its partial :accum / :reduced buffers
    function run_rank(r, nr)
        bank = bb_bank(kernel, ints, values, seed)
        stacks = STADE.val_init_stacks(initstacks_fn, Any[haskey(ints, n) ? ints[n] : deepcopy(values[n]) for n in stack_names])
        Base.invokelatest(reset_accum!, [bank[b] for b in accum]...); Base.invokelatest(reset_reduced!, [bank[b] for b in reduced]...)
        for s in 1:nsamples
            mod(s - 1, nr) == r || continue
            copyto!(bank[:x], xs[s])
            Base.invokelatest(reset_local!, [bank[b] for b in local_bufs]...)
            for d in sig.dependents
                sig.kinds[d] == :array_float || continue
                b = STADE.agen_shadow(d); roles[b] in (:accum, :reduced) && continue
                bank[b] .= 1.0
            end
            args = Any[bank[b] for b in STADE.agen_signature_args(sig)]; append!(args, collect(stacks))
            Base.invokelatest(adjoint_fn, args...)
        end
        return Dict(b => bank[b] for b in vcat(accum, reduced))
    end
    sumover(parts) = Dict(b => sum(p[b] for p in parts) for b in keys(parts[1]))

    run_rank(0, 1)                                           # compile
    t_ref = minimum(@elapsed(run_rank(0, 1)) for _ in 1:reps)
    ref = run_rank(0, 1)
    t_one = t_ref / nsamples
    local t_par, got
    if comm === nothing || P == 1
        t_par = minimum(@elapsed(sumover([run_rank(r, serial_ranks) for r in 0:serial_ranks-1])) for _ in 1:reps)   # sum of both ranks' times: not a parallel time
        got = sumover([run_rank(r, serial_ranks) for r in 0:serial_ranks-1])
    else
        function par()
            part = run_rank(rank, P)
            for b in sort(collect(keys(part))); MPI.Allreduce!(part[b], +, comm); end
            return part
        end
        par(); MPI.Barrier(comm)
        ts = Float64[]
        for _ in 1:reps
            MPI.Barrier(comm); t0 = time(); got = par(); MPI.Barrier(comm); push!(ts, time() - t0)
        end
        t_par = minimum(ts)
    end
    worst = maximum(maximum(abs, got[b] .- ref[b]) / max(maximum(abs, ref[b]), 1.0) for b in keys(ref))
    ok = worst <= rtol
    if rank == 0
        println("kernel ", basename(path), ": ", nsamples, " samples, ranks ", comm === nothing || P == 1 ? "$(serial_ranks) (serial stand-in)" : string(P), ", accum buffers ", length(accum))
        println(rpad(ok ? "ok" : "FAIL", 6), "batch gradient equals the one-rank loop, max_rel_err=", round(worst, sigdigits = 3))
        println("one sample, adjoint:           ", round(1e3 * t_one, digits = 2), " ms")
        println("all samples, one rank:         ", round(1e3 * t_ref, digits = 2), " ms")
        println(comm === nothing || P == 1 ? "all samples, ranks in series:  " : "all samples, $(P) ranks + Allreduce: ", round(1e3 * t_par, digits = 2), " ms",
                comm === nothing || P == 1 ? "  (not a parallel time)" : "  speedup " * string(round(t_ref / t_par, digits = 2)))
    end
    return (status = ok ? 0 : 1, rank = rank, ranks = P, max_rel_err = worst, t_one_ms = 1e3 * t_one, t_one_rank_ms = 1e3 * t_ref, t_par_ms = 1e3 * t_par)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(bench_batch(isempty(ARGS) ? bb_default_kernel() : ARGS[1]).status)
end
