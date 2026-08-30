# ==================== validate_corpus_gpu.jl ========================
# GPU counterpart to validate_corpus.jl. It walks the same corpus and
# follows the same "differentiate every primal, then check the result"
# shape, but where validate_corpus.jl checks the CPU tangent, adjoint,
# HVP, and dot-product paths, this script checks CUDA and JACC device
# parity, LIVE, on the GPU attached to the machine that runs it.
#
# This script does not call STADE.stade_validate_gpu_file or
# STADE.stade_jacc_file. Those functions exist because the sandbox
# this codebase was developed in has no GPU, so they only emit a
# self-contained script for someone else to run elsewhere (for example
# through the runpod-julia-cuda-jacc skill's relay). This script
# assumes the opposite: it runs somewhere with a real CUDA device
# already attached (a local workstation, a cloud VM you are logged
# into), and with JACC's backend already set to CUDA. It never writes
# a script for later submission and never touches RunPod. It builds
# real CuArray/JACC.array values, calls the generated functions with
# Base.invokelatest, and compares the results in-process.
#
# Flow per corpus kernel:
#   1. Differentiate the monoprocessor primal -- STADE.stade_adjoint on
#      the plain, single-threaded kernel source, with keep_push_pop =
#      false. This is the only mode with a GPU target at all, since
#      :stack mode's push!/pop! calls are host-only by nature.
#   2. Run GPU codegen -- STADE.stade_gpu(..., STADE.cgen_backend_cuda())
#      for CUDA, STADE.stade_jacc(...) for JACC, on both the adjoint
#      and its initstacks function.
#   3. Execute: evaluate every generated device kernel and host
#      function into Main, run the CPU adjoint and the device host
#      function against the SAME random baseline and the SAME
#      per-trial reverse-mode seed, then compare every mutated array
#      and returned scalar element-wise.
#
# Usage: run `julia validate_corpus_gpu.jl` from this directory, with
# the corpus directory (default "val-corpus") alongside it. Or include
# this file and call `validate_corpus_gpu(dir; kwargs...)` directly for
# a specific subset -- pass `only = ["name1", "name2"]` to select
# kernels by name.

include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# The generated code (both CUDA's and JACC's) refers to CUDA.*, JACC.*,
# and Atomix.* by name. Core.eval'ing it into Main needs these already
# bound there, exactly like the preamble STADE.cgen_backend_cuda's
# `preamble` field and STADE.jgen_preamble() build for a remotely
# submitted script. The `using`/`import` lines below are the live-
# execution equivalent of that preamble text.
using CUDA
import JACC
JACC.@init_backend
import Atomix

# Matches STADE.cgen_backend_cuda's own preamble
# (`CUDA.allowscalar(false)`): any accidental scalar touch on a
# CuArray must fail here, not succeed slowly, exactly as it would for
# a remotely submitted script built from the same preamble.
CUDA.allowscalar(false)

# ---- live-execution helpers (mirror STADE.val_gpu_parity_script's
#      text-generation helpers, but with real values instead of
#      literal source) --------------------------------------------

# Mirrors _val_init_stacks_local in every script STADE.stade_validate_gpu_file
# or STADE.stade_validate_jacc_file emits: a Tier A / no-stack kernel's
# initstacks_* function returns `nothing` (call it with zero stacks), a
# single-stack kernel returns a bare value (not a tuple), and every
# other kernel returns a tuple already in call order.
function gval_live_init_stacks(fn, extra_args)
    r = Base.invokelatest(fn, extra_args...)
    r === nothing && return ()
    r isa Tuple && return r
    return (r,)
end

# A stack_arg_name is either a scalar_int kernel argument (shared
# verbatim, unwrapped, between the CPU and device calls) or its own
# entry in a values Dict -- and that entry can itself be a scalar or
# an array, so `wrap` applies only when the entry is an array. `wrap`
# is `identity` on the CPU side, and CuArray or JACC.array on the
# device side.
function gval_stack_extra(names::Vector{Symbol}, int_args::Dict, values::Dict, wrap)
    return Any[haskey(int_args, n) ? int_args[n] :
               (values[n] isa AbstractArray ? wrap(deepcopy(values[n])) : values[n])
               for n in names]
end

# Builds one call's positional argument list and a name-keyed dict of
# the actual mutable array objects passed in, so the caller can
# inspect them again after the call without reconstructing anything --
# a Julia Array, CuArray, or JACC.array is passed by reference, so
# whatever the callee wrote into it is visible through the SAME object
# already held here. `wrap` is `identity` for the CPU call and
# CuArray/JACC.array for the device call. Everything else about the
# argument order is backend-independent (see STADE.val_gpu_call_args's
# matching order): scalar_int arguments pass bare, scalar_float and
# array_float arguments pass as a (value, shadow) pair, and array_int
# arguments pass as a bare value only, since STADE never differentiates
# them.
function gval_build_call(sig, int_args::Dict, values::Dict, seed::Dict, wrap;
                         mode::Symbol = :adjoint,
                         dvalues::Dict = Dict{Symbol,Any}(), dseed::Dict = Dict{Symbol,Any}())
    args = Any[]
    tracked = Dict{Symbol,Any}()
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(args, int_args[a])
        elseif k == :scalar_float
            push!(args, values[a])
            push!(args, seed[a])
        elseif k == :array_float
            v = wrap(deepcopy(values[a]))
            s = wrap(deepcopy(seed[a]))
            push!(args, v)
            push!(args, s)
            tracked[a] = (kind = :array_float, val = v, sh = s)
        else # :array_int
            v = wrap(deepcopy(values[a]))
            push!(args, v)
            tracked[a] = (kind = :array_int, val = v)
        end
    end
    # hvp_ appends one tangent pair per float arg AFTER the whole adjoint list,
    # in sig.args order -- the same second pass STADE.val_hvp_call_args makes.
    # Getting this order wrong silently transposes two device arrays of equal
    # length rather than raising, so it has to mirror the generator exactly.
    if mode == :hvp
        for a in sig.args
            k = sig.kinds[a]
            k in (:scalar_float, :array_float) || continue
            if k == :scalar_float
                push!(args, dvalues[a])
                push!(args, dseed[a])
            else
                t = wrap(deepcopy(dvalues[a]))
                st = wrap(deepcopy(dseed[a]))
                push!(args, t)
                push!(args, st)
                tracked[a] = merge(tracked[a], (tan = t, shtan = st))
            end
        end
    end
    return args, tracked
end

gval_relerr_scalar(a, b) = abs(a - b) / max(abs(a), abs(b), 1.0)
# `init = 0.0`: the baseline generator gates a draw only on initstacks_* running,
# so an all-zero integer draw is dimensionally coherent and accepted, giving
# zero-length arrays. `maximum` over those throws instead of comparing nothing.
gval_relerr_arr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1.0); init = 0.0)

# `dev_to_host` brings a device array back before the comparison --
# `Array` for CUDA, `JACC.to_host` for JACC. This checks every mutated
# float array's shadow (the adjoint itself, the actual quantity of
# interest) and its value (mutated in place by the forward
# re-execution keep_push_pop=false checkpoints, so it should also be
# bit-identical modulo the device's own floating-point reordering).
# For a read-only array_int table it also checks the value, which
# should never differ at all -- a cheap extra check.
# Returns (worst_relerr, elements_changed). What separates a real pass from a
# vacuous one is elements the CPU reference actually WROTE, not elements
# compared: an all-zero integer draw leaves every array at its baseline length
# while collapsing every loop bound to zero trips, so a length-based count reads
# healthy (measured live at 416) for a run that executed no device code at all.
# `pre` holds a copy of each CPU array taken immediately before the call.
function gval_compare(cpu_tracked::Dict, dev_tracked::Dict, dev_to_host, pre::Dict)
    m = 0.0
    changed = 0
    nchanged(x, y) = count(!=(0.0), x .- y)
    for (a, ct) in cpu_tracked
        dt = dev_tracked[a]
        m = max(m, gval_relerr_arr(ct.val, dev_to_host(dt.val)))
        changed += nchanged(ct.val, pre[a].val)
        if ct.kind == :array_float
            m = max(m, gval_relerr_arr(ct.sh, dev_to_host(dt.sh)))
            changed += nchanged(ct.sh, pre[a].sh)
        end
        if hasproperty(ct, :tan)
            m = max(m, gval_relerr_arr(ct.tan, dev_to_host(dt.tan)))
            m = max(m, gval_relerr_arr(ct.shtan, dev_to_host(dt.shtan)))
            changed += nchanged(ct.tan, pre[a].tan) + nchanged(ct.shtan, pre[a].shtan)
        end
    end
    return m, changed
end

# Runs `trials` reverse-mode seeds through both the CPU adjoint and one
# device plan (CUDA's or JACC's, whichever (wrap, dev_to_host,
# atomic_macro) triple is passed in), evaluating every kernel and host
# function into Main first. Returns the same (ok, max_rel_err) shape
# that the verdict from STADE.stade_validate_gpu_file or
# STADE.stade_validate_jacc_file uses, computed here in-process instead
# of parsed back from a pushed result.
function gval_check_backend(kernel, int_args::Dict, values::Dict, seeds::Vector,
                             gen_out, plan_init, plan_gen, stack_arg_names::Vector{Symbol},
                             wrap, dev_to_host, atomic_macro;
                             mode::Symbol = :adjoint,
                             tan_seeds::Vector = Any[], shtan_seeds::Vector = Any[])
    cpu_gen_expr = mode == :hvp ? gen_out.hvp : gen_out.adjoint
    for k in plan_init.kernels
        Core.eval(Main, k)
    end
    Core.eval(Main, plan_init.host)
    for k in plan_gen.kernels
        Core.eval(Main, k)
    end
    Core.eval(Main, plan_gen.host)

    # The CPU reference pair has to be DEFINED here too, not just named. Evaluating
    # `adjoint_out.adjoint.args[1].args[1]` alone resolves the function's name and nothing
    # else, so without these two lines every kernel fails with `UndefVarError: <name>_b not
    # defined in Main` -- the device side was being compared against a function that was never
    # brought into scope. Confirmed on a live GPU: this is what made the whole script report
    # gen_error for all 41 corpus kernels.
    Core.eval(Main, gen_out.initstacks)
    Core.eval(Main, cpu_gen_expr)

    cpu_init_fn = Core.eval(Main, gen_out.initstacks.args[1].args[1])
    cpu_gen_fn  = Core.eval(Main, cpu_gen_expr.args[1].args[1])
    dev_init_fn = Core.eval(Main, plan_init.host.args[1].args[1])
    dev_gen_fn  = Core.eval(Main, plan_gen.host.args[1].args[1])

    sig = kernel.sig
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    # Same non-associative-atomic-summation reasoning as the resolved_rtol in
    # STADE.stade_validate_gpu_file: scale the default tolerance by the number of
    # @atomic/Atomix.@atomic sites the generated adjoint actually has.
    atomic_sites = sum(STADE.val_count_atomic_sites(k, atomic_macro) for k in plan_gen.kernels; init = 0)
    rtol = 2.5e-14 * max(1, atomic_sites)
    # each scalar_float arg returns one value from an adjoint and two (ab, abd)
    # from an HVP, so emit_return_scalars only ever yields a bare non-tuple in
    # the adjoint's single-scalar case
    n_ret = mode == :hvp ? 2 * length(scalar_args) : length(scalar_args)

    max_err = 0.0
    n_changed = 0
    for (i_seed, seed) in enumerate(seeds)
        dvals = mode == :hvp ? tan_seeds[i_seed] : Dict{Symbol,Any}()
        dsd   = mode == :hvp ? shtan_seeds[i_seed] : Dict{Symbol,Any}()
        cpu_args, cpu_tracked = gval_build_call(sig, int_args, values, seed, identity; mode, dvalues = dvals, dseed = dsd)
        dev_args, dev_tracked = gval_build_call(sig, int_args, values, seed, wrap; mode, dvalues = dvals, dseed = dsd)
        stacks_cpu = gval_live_init_stacks(cpu_init_fn, gval_stack_extra(stack_arg_names, int_args, values, identity))
        stacks_dev = gval_live_init_stacks(dev_init_fn, gval_stack_extra(stack_arg_names, int_args, values, wrap))

        # snapshot the CPU arrays before the call, so gval_compare can tell how much
        # the reference actually wrote
        pre = Dict{Symbol,Any}(a => (ct.kind == :array_int ? (val = deepcopy(ct.val),) :
                                     hasproperty(ct, :tan) ?
                                        (val = deepcopy(ct.val), sh = deepcopy(ct.sh),
                                         tan = deepcopy(ct.tan), shtan = deepcopy(ct.shtan)) :
                                        (val = deepcopy(ct.val), sh = deepcopy(ct.sh)))
                               for (a, ct) in cpu_tracked)
        ret_cpu = Base.invokelatest(cpu_gen_fn, cpu_args..., stacks_cpu...)
        ret_dev = Base.invokelatest(dev_gen_fn, dev_args..., stacks_dev...)

        m, n = gval_compare(cpu_tracked, dev_tracked, dev_to_host, pre)
        n_changed += n
        if n_ret > 0
            ret_cpu_t = n_ret == 1 ? (ret_cpu,) : ret_cpu
            ret_dev_t = n_ret == 1 ? (ret_dev,) : ret_dev
            for i in 1:n_ret
                m = max(m, gval_relerr_scalar(ret_cpu_t[i], ret_dev_t[i]))
            end
        end
        max_err = max(max_err, m)
    end
    return (ok = max_err <= rtol && n_changed > 0, max_rel_err = max_err, rtol = rtol,
            elements_changed = n_changed, vacuous = n_changed == 0)
end

# ---- top-level driver ------------------------------------------------

function validate_corpus_gpu(dir::String = "val-corpus"; trials::Int = 3,
                              fuse_ii_loops::Bool = false,
                              run_cuda::Bool = true, run_jacc::Bool = true,
                              write_generated::Bool = true,
                              modes::Vector{Symbol} = [:adjoint, :hvp],
                              only::Union{Vector{String},Nothing} = nothing)
    for f in readdir(dir)
        if endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl") ||
           endswith(f, ".yaml") || endswith(f, ".gpu.yaml") ||
           endswith(f, "_cuda.jl") || endswith(f, "_jacc.jl")
            rm(joinpath(dir, f))
        end
    end
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl") ||
                    endswith(f, "_cuda.jl") || endswith(f, "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))

    cuda_backend = STADE.cgen_backend_cuda()
    jacc_atomic_macro = Expr(:., :Atomix, QuoteNode(Symbol("@atomic")))

    results = NamedTuple[]
    for name in names
        path = joinpath(dir, name * ".jl")
        # own namespace, not the CPU oracles' shared `.yaml` -- validate_zerotrip.jl
        # draws every integer from [0, 2] and leaves its baselines behind, and a GPU
        # run inheriting that draw executes no device code and reports a pass
        yp = STADE.io_gpu_yaml_path(path)
        isfile(yp) || STADE.stade_generate_baseline_file(path; yaml_path = yp)
        primal_expr = STADE.io_read_corpus_entry(path)
        kernel = STADE.parse_kernel(primal_expr)
        baseline = STADE.io_read_baseline_yaml(yp)
        STADE.val_coerce_int_arrays!(kernel, baseline.values)

        for mode in modes
        # ---- step 1: differentiate the monoprocessor primal ---------
        local gen_out
        try
            gen_out = mode == :hvp ? STADE.stade_hvp(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops) :
                                     STADE.stade_adjoint(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops)
        catch e
            println("  !! ", name, " [", mode, "] ", first(split(sprint(showerror, e), "\n")))
            push!(results, (kernel = name, mode = mode, backend = :gen, status = :gen_error, max_rel_err = NaN, elements_changed = 0))
            continue
        end
        gen_expr = mode == :hvp ? gen_out.hvp : gen_out.adjoint
        suffix = mode == :hvp ? "_hv" : "_b"
        if write_generated
            open(joinpath(dir, name * suffix * ".jl"), "w") do f
                write(f, STADE.io_expr_to_source(gen_out.initstacks))
                write(f, STADE.io_expr_to_source(gen_expr))
            end
        end
        stack_arg_names = Symbol[a for a in gen_out.initstacks.args[1].args[2:end]]
        seeds = [STADE.val_random_values_like(kernel, baseline.values) for _ in 1:trials]
        # the HVP's two extra per-trial directions, drawn independently of the
        # reverse seed and of each other
        tan_seeds   = mode == :hvp ? [STADE.val_random_values_like(kernel, baseline.values) for _ in 1:trials] : Any[]
        shtan_seeds = mode == :hvp ? [STADE.val_random_values_like(kernel, baseline.values) for _ in 1:trials] : Any[]

        # ---- step 2 + 3: CUDA codegen, then execute ------------------
        if run_cuda
            try
                plan_init = STADE.stade_gpu(gen_out.initstacks, cuda_backend)
                plan_gen  = STADE.stade_gpu(gen_expr, cuda_backend)
                if write_generated
                    open(joinpath(dir, name * suffix * "_cuda.jl"), "w") do f
                        for k in plan_init.kernels
                            write(f, STADE.io_expr_to_source(k))
                        end
                        write(f, STADE.io_expr_to_source(plan_init.host))
                        for k in plan_gen.kernels
                            write(f, STADE.io_expr_to_source(k))
                        end
                        write(f, STADE.io_expr_to_source(plan_gen.host))
                    end
                end
                r = gval_check_backend(kernel, baseline.int_args, baseline.values, seeds,
                                        gen_out, plan_init, plan_gen, stack_arg_names,
                                        CuArray, Array, cuda_backend.atomic_macro;
                                        mode, tan_seeds, shtan_seeds)
                push!(results, (kernel = name, mode = mode, backend = :cuda, status = r.ok ? :ok : (r.vacuous ? :VACUOUS : :FAIL), max_rel_err = r.max_rel_err, elements_changed = r.elements_changed))
            catch e
                println("  !! ", name, " [", mode, "/cuda] ", first(split(sprint(showerror, e), "\n")))
                push!(results, (kernel = name, mode = mode, backend = :cuda, status = :gen_error, max_rel_err = NaN, elements_changed = 0))
            end
        end

        # ---- step 2 + 3: JACC codegen, then execute ------------------
        if run_jacc
            try
                plan_init = STADE.stade_jacc(gen_out.initstacks)
                plan_gen  = STADE.stade_jacc(gen_expr)
                if write_generated
                    open(joinpath(dir, name * suffix * "_jacc.jl"), "w") do f
                        for k in plan_init.kernels
                            write(f, STADE.io_expr_to_source(k))
                        end
                        write(f, STADE.io_expr_to_source(plan_init.host))
                        for k in plan_gen.kernels
                            write(f, STADE.io_expr_to_source(k))
                        end
                        write(f, STADE.io_expr_to_source(plan_gen.host))
                    end
                end
                r = gval_check_backend(kernel, baseline.int_args, baseline.values, seeds,
                                        gen_out, plan_init, plan_gen, stack_arg_names,
                                        x -> JACC.array(x), x -> JACC.to_host(x), jacc_atomic_macro;
                                        mode, tan_seeds, shtan_seeds)
                push!(results, (kernel = name, mode = mode, backend = :jacc, status = r.ok ? :ok : (r.vacuous ? :VACUOUS : :FAIL), max_rel_err = r.max_rel_err, elements_changed = r.elements_changed))
            catch e
                println("  !! ", name, " [", mode, "/jacc] ", first(split(sprint(showerror, e), "\n")))
                push!(results, (kernel = name, mode = mode, backend = :jacc, status = :gen_error, max_rel_err = NaN, elements_changed = 0))
            end
        end
        end
    end

    for r in results
        println(rpad("$(r.kernel) [$(r.mode)/$(r.backend)]", 38), " ", r.status,
                isnan(r.max_rel_err) ? "" : "  max_rel_err=$(round(r.max_rel_err, sigdigits = 4))",
                "  changed=$(r.elements_changed)")
    end
    bad = count(r -> r.status != :ok, results)
    println("\n", length(results) - bad, "/", length(results), " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return results
end

validate_corpus_gpu()