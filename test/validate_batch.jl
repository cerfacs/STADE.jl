include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# Serial communicator. Every rank runs in this process, one after
# another, and this stands in where MPI.Allreduce! would run. It needs no
# MPI, so the whole rank-invariance oracle runs in a sandbox with no
# cluster. It does not test CUDA-aware transport; that is the cluster's
# only job.
#
# The reset functions are called as generated, not reimplemented. They
# contain no MPI, and section 2.2 puts the entire per-sample zeroing
# contract inside bgen_reset_local!, so a reimplementation here would
# test this file instead of the stage.

"""
    batch_bank(kernel, int_args, values, seed, stacks)

One dictionary holding every buffer of the adjoint by name, allocated
once and reused across samples. `val_adjoint_call_args` deepcopies, which
is right for a single call and wrong here: the whole point is that the
`:accum` buffers persist while the `:local` ones are cleared.
"""
function batch_bank(kernel, int_args::Dict, values::Dict, seed::Dict)
    sig = kernel.sig
    bank = Dict{Symbol,Any}()
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            bank[a] = int_args[a]
        elseif k in (:scalar_float, :array_float)
            bank[a] = deepcopy(values[a])
            bank[STADE.agen_shadow(a)] = deepcopy(seed[a])
        else
            bank[a] = deepcopy(values[a])
        end
    end
    return bank
end

# val_init_stacks returns a Tuple, empty when the kernel has no stack.
# vcat would splice an empty tuple in as one argument, so append! it.
function batch_call_args(kernel, bank, stacks)
    call = Any[bank[b] for b in STADE.agen_signature_args(kernel.sig)]
    append!(call, collect(stacks))
    return call
end

"""
    batch_run(...)

Run `nranks` ranks over `nsamples` samples and return the summed
`:accum` and `:reduced` buffers. Sample `s` goes to rank `mod(s-1, nranks)`,
so a rank count that does not divide the sample count gives ranks
different local counts, which is the case section 9.4 covers.
"""
function batch_run(kernel, adjoint_fn, initstacks_fn, stack_names, roles,
                   reset_accum!, reset_reduced!, reset_local!,
                   int_args, sample_values, shared_values, seed, nranks::Int)
    sig = kernel.sig
    accum   = STADE.bgen_role_args(kernel, roles, :adjoint, :accum)
    reduced = STADE.bgen_role_args(kernel, roles, :adjoint, :reduced)
    total = Dict{Symbol,Any}()
    for r in 0:(nranks - 1)
        mine = [s for s in 1:length(sample_values) if mod(s - 1, nranks) == r]
        values = merge(shared_values, isempty(mine) ? sample_values[1] : sample_values[mine[1]])
        bank = batch_bank(kernel, int_args, values, seed)
        stack_extra = [haskey(int_args, n) ? int_args[n] : deepcopy(values[n]) for n in stack_names]
        stacks = STADE.val_init_stacks(initstacks_fn, stack_extra)
        Base.invokelatest(reset_accum!, [bank[b] for b in accum]...)
        Base.invokelatest(reset_reduced!, [bank[b] for b in reduced]...)
        for s in mine
            for (k, v) in sample_values[s]
                bank[k] = v isa Number ? v : copy(v)
            end
            local_bufs = STADE.bgen_role_args(kernel, roles, :adjoint, :local)
            Base.invokelatest(reset_local!, [bank[b] for b in local_bufs]...)
            # Seed only the buffers the caller owns. Writing every shadow
            # here would clear the :accum and :reduced buffers on each
            # sample, leaving the last sample's contribution alone.
            for d in sig.dependents
                sig.kinds[d] == :array_float || continue
                b = STADE.agen_shadow(d)
                roles[b] in (:accum, :reduced) && continue
                bank[b] .= 1.0
            end
            Base.invokelatest(adjoint_fn, batch_call_args(kernel, bank, stacks)...)
        end
        # the serial stand-in for the two MPI.Allreduce! calls
        for b in vcat(accum, reduced)
            total[b] = haskey(total, b) ? total[b] .+ bank[b] : copy(bank[b])
        end
    end
    return total
end

function validate_batch(; dir::String = joinpath(@__DIR__, "val-corpus"),
                          nsamples::Int = 6, rtol::Float64 = 1e-12)
    # Declarations are per kernel and carry the one fact the stage cannot
    # derive: which read-only arrays change from one sample to the next.
    # For the ML kernels that is the input; every weight and gain array
    # falls to :shared. For ttgc the mesh geometry is shared across the
    # ensemble and only the state fields vary, and res/res2 are the
    # ambiguous shape resolved as per-sample residuals.
    subjects = [("ddp_affine", [:u], Symbol[]),
                ("ddp_twoparam", [:u], Symbol[]),
                ("ddp_ragged", [:u], Symbol[]),
                ("ddp_branch", [:u], Symbol[]),
                ("ddp_tangent_input", [:u], Symbol[]),
                ("ddp_batch_hist", [:u], [:hist]),
                ("affine_loss", [:u], Symbol[]),
                ("matvec_loss", [:u, :v], Symbol[]),
                ("mpnn", [:node_feat, :edge_feat], Symbol[]),
                ("ttgc", [:u, :uref, :res, :res2], Symbol[]),
                # transformer updates x in place across layers, so x is
                # written and derives :scratch on its own. Naming it here
                # is refused, and the derived role is already the right
                # one: xb becomes :local, per-rank and zeroed per sample.
                ("transformer", Symbol[], Symbol[]),
                ("unet", [:x], Symbol[])]
    isempty(ARGS) || (subjects = [s for s in subjects if s[1] in ARGS])
    checks = 0; bad = 0
    for (name, per_sample, reduced) in subjects
        Random.seed!(hash(name))
        primal = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        kernel = STADE.parse_kernel(primal)
        # The baseline MUST be generated before the adjoint is compiled.
        # val_generate_baseline compiles its own initstacks_* with
        # fuse_ii_loops defaulting to false, which redefines the name in
        # Main. Compiling the fused one first lets the baseline clobber it,
        # and ttgc then gets 13 stacks passed to an adjoint expecting 4.
        # The small kernels hide this: both flag values give them the same
        # stack count.
        baseline = STADE.val_generate_baseline(kernel, primal)

        out = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
        stack_names = Symbol[a for a in out.initstacks.args[1].args[2:end]]
        adjoint_fn = STADE.val_compile(out.adjoint)
        initstacks_fn = STADE.val_compile(out.initstacks)

        ep = STADE.stade_batch(primal; per_sample = per_sample, mode = :adjoint, reduced = reduced)
        reset_accum!   = STADE.val_compile(ep.reset_accum.expr)
        reset_reduced! = STADE.val_compile(ep.reset_reduced.expr)
        reset_local!   = STADE.val_compile(ep.reset_local.expr)
        int_args = baseline.int_args
        # :shared and :scratch keep one value across samples; :per_sample
        # gets a fresh draw per sample, which is what makes a missing
        # collective visible at all
        ps_set = Set(per_sample)
        shared_values = Dict(k => v for (k, v) in baseline.values if !(k in ps_set))
        sample_values = [Dict(k => STADE.val_random_values_like(kernel, baseline.values)[k]
                              for k in per_sample) for _ in 1:nsamples]
        seed = STADE.val_zeros_like(kernel, baseline.values)
        for d in kernel.sig.dependents
            kernel.sig.kinds[d] == :array_float && (seed[d] = ones(size(baseline.values[d])))
        end

        ref = batch_run(kernel, adjoint_fn, initstacks_fn, stack_names, ep.roles,
                        reset_accum!, reset_reduced!, reset_local!,
                        int_args, sample_values, shared_values, seed, 1)
        for P in (2, 3, 4)
            got = batch_run(kernel, adjoint_fn, initstacks_fn, stack_names, ep.roles,
                            reset_accum!, reset_reduced!, reset_local!,
                            int_args, sample_values, shared_values, seed, P)
            checks += 1
            worst = 0.0; worst_b = :none
            for b in sort(collect(keys(ref)))
                den = max(maximum(abs, ref[b]), 1.0)
                e = maximum(abs, got[b] .- ref[b]) / den
                e > worst && (worst = e; worst_b = b)
            end
            if worst <= rtol
                println(rpad("$name [rank invariance P=$P]", 42), " ok  max_rel_err=",
                        round(worst, sigdigits = 4))
            else
                bad += 1
                println(rpad("$name [rank invariance P=$P]", 42), " FAIL  ", worst_b,
                        " max_rel_err=", worst)
            end
        end

        # Delete-one-collective: a buffer whose rank-0 partial already
        # equals the reference never needed summing, and one that differs
        # proves the collective did real work. Section 6.3.
        checks += 1
        r0 = batch_run(kernel, adjoint_fn, initstacks_fn, stack_names, ep.roles,
                       reset_accum!, reset_reduced!, reset_local!,
                       int_args, sample_values, shared_values, seed, 1)
        r0_partial = Dict{Symbol,Any}()
        begin
            accum = STADE.bgen_role_args(kernel, ep.roles, :adjoint, :accum)
            reduced_b = STADE.bgen_role_args(kernel, ep.roles, :adjoint, :reduced)
            mine = [s for s in 1:nsamples if mod(s - 1, 2) == 0]
            values = merge(shared_values, sample_values[mine[1]])
            bank = batch_bank(kernel, int_args, values, seed)
            stack_extra = [haskey(int_args, n) ? int_args[n] : deepcopy(values[n]) for n in stack_names]
            stacks = STADE.val_init_stacks(initstacks_fn, stack_extra)
            Base.invokelatest(reset_accum!, [bank[b] for b in accum]...)
            Base.invokelatest(reset_reduced!, [bank[b] for b in reduced_b]...)
            for s in mine
                for (k, v) in sample_values[s]
                    bank[k] = v isa Number ? v : copy(v)
                end
                Base.invokelatest(reset_local!, [bank[b] for b in STADE.bgen_role_args(kernel, ep.roles, :adjoint, :local)]...)
                for d in kernel.sig.dependents
                    kernel.sig.kinds[d] == :array_float || continue
                    b = STADE.agen_shadow(d)
                    ep.roles[b] in (:accum, :reduced) && continue
                    bank[b] .= 1.0
                end
                Base.invokelatest(adjoint_fn, batch_call_args(kernel, bank, stacks)...)
            end
            for b in vcat(accum, reduced_b)
                r0_partial[b] = copy(bank[b])
            end
        end
        inert = [b for b in sort(collect(keys(r0))) if maximum(abs, r0_partial[b] .- r0[b]) == 0.0]
        if isempty(inert)
            println(rpad("$name [collectives do work]", 42), " ok  ",
                    length(keys(r0)), " buffers all change")
        else
            bad += 1
            println(rpad("$name [collectives do work]", 42), " FAIL  ", inert,
                    " identical without summing")
        end
    end
    # ---- accumulation window: two roles, two periods ----
    # W micro-batches accumulate :accum locally and sum once at the window
    # boundary, while :reduced is summed every micro-batch. The window
    # gradient must equal the gradient of one batch holding every sample.
    # Nothing else reaches this: with one period the two are the same run.
    for (name, per_sample, reduced) in subjects
        name in ("ddp_affine", "ddp_batch_hist") || continue
        Random.seed!(hash(name * "window"))
        primal = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        kernel = STADE.parse_kernel(primal)
        baseline = STADE.val_generate_baseline(kernel, primal)
        out = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
        stack_names = Symbol[a for a in out.initstacks.args[1].args[2:end]]
        adjoint_fn = STADE.val_compile(out.adjoint)
        initstacks_fn = STADE.val_compile(out.initstacks)
        ep = STADE.stade_batch(primal; per_sample = per_sample, mode = :adjoint, reduced = reduced)
        ra = STADE.val_compile(ep.reset_accum.expr)
        rr = STADE.val_compile(ep.reset_reduced.expr)
        rl = STADE.val_compile(ep.reset_local.expr)

        ps_set = Set(per_sample)
        shared_values = Dict(k => v for (k, v) in baseline.values if !(k in ps_set))
        sv = [Dict(k => STADE.val_random_values_like(kernel, baseline.values)[k]
                   for k in per_sample) for _ in 1:6]
        seed = STADE.val_zeros_like(kernel, baseline.values)

        # one batch of 6 samples, gradients summed once
        one = batch_run(kernel, adjoint_fn, initstacks_fn, stack_names, ep.roles,
                        ra, rr, rl, baseline.int_args, sv, shared_values, seed, 2)
        # three windows of 2 samples: reset :accum only at the start, sum
        # only at the end, which is what the two collectives allow
        accum = STADE.bgen_role_args(kernel, ep.roles, :adjoint, :accum; arrays_only = true)
        win = Dict{Symbol,Any}()
        for w in 1:3
            part = batch_run(kernel, adjoint_fn, initstacks_fn, stack_names, ep.roles,
                             ra, rr, rl, baseline.int_args, sv[(2w - 1):(2w)],
                             shared_values, seed, 2)
            for b in accum
                win[b] = haskey(win, b) ? win[b] .+ part[b] : copy(part[b])
            end
        end
        checks += 1
        worst = 0.0
        for b in accum
            den = max(maximum(abs, one[b]), 1.0)
            worst = max(worst, maximum(abs, win[b] .- one[b]) / den)
        end
        if worst <= rtol
            println(rpad("$name [accumulation window]", 42), " ok  max_rel_err=",
                    round(worst, sigdigits = 4))
        else
            bad += 1
            println(rpad("$name [accumulation window]", 42), " FAIL  max_rel_err=", worst)
        end
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_batch()
