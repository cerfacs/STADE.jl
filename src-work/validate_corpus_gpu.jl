# ==================== validate_corpus_gpu.jl ========================
# GPU analog of validate_corpus.jl. Same corpus, same "differentiate
# every primal, then check the result" shape -- but where
# validate_corpus.jl checks the CPU tangent/adjoint/hvp/dotprod paths,
# this checks CUDA and JACC device parity, LIVE, on whatever GPU is
# actually attached to the machine running this script.
#
# This is NOT stade_validate_gpu_file/stade_validate_jacc_file (see
# STADE.jl's own Item 1/Item 4 comments): those exist because the
# Claude sandbox this codebase was developed in has no GPU at all, so
# they only ever emit a self-contained script for someone else to run
# elsewhere (e.g. via the runpod-julia-cuda-jacc skill's relay). This
# script assumes the OPPOSITE: it's running somewhere with a real CUDA
# device already attached (a local workstation, a cloud VM you're
# ssh'd into, etc.), and JACC's backend already set to CUDA. It never
# writes a script for later submission and never touches RunPod at
# all -- it builds real CuArray/JACC.array values, calls the real
# generated functions with Base.invokelatest, and compares in-process.
#
# Flow per corpus kernel, matching the plan's own phrasing:
#   1. differentiate the monoprocessor primal -- stade_adjoint on the
#      plain, single-threaded kernel source, keep_push_pop=false (the
#      only mode with a GPU target at all: :stack mode's push!/pop! is
#      inherently host-only).
#   2. GPU codegen -- stade_gpu(..., cgen_backend_cuda()) for CUDA,
#      stade_jacc(...) for JACC, on both the adjoint and its
#      initstacks.
#   3. execute -- eval every generated device kernel + host function
#      into Main, run the CPU adjoint and the device host function
#      against the SAME random baseline (and the SAME per-trial
#      reverse-mode seed), and compare every mutated array and
#      returned scalar element-wise.
#
# Usage: `julia validate_corpus_gpu.jl` from the directory containing
# this file and the corpus directory (default "val-corpus"), or
# `include("validate_corpus_gpu.jl")` then call
# `validate_corpus_gpu(dir; kwargs...)` directly for a specific subset
# (validate_corpus_gpu_sel.jl-style kernel-name filtering can be added
# the same way validate_corpus.jl's own _sel variants work, by passing
# `only = ["name1", "name2"]`).

include("STADE.jl")

# The generated code (both CUDA's and JACC's) references CUDA.*/
# JACC.*/Atomix.* by name -- Core.eval'ing it into Main requires these
# already bound there, exactly like stade_gpu_plan.md's own generated-
# script preambles (see cgen_backend_cuda()'s `preamble` field and
# jgen_preamble()) do for a REMOTELY-submitted script. `using` here is
# the live-execution equivalent of that preamble text.
using CUDA
import JACC
JACC.@init_backend
import Atomix

# Matches cgen_backend_cuda()'s own preamble (`CUDA.allowscalar(false)`):
# any accidental scalar touch on a CuArray should fail loudly here,
# not silently succeed slowly, exactly as it would for a remotely
# submitted script built from the same preamble.
CUDA.allowscalar(false)

# ---- live-execution helpers (the in-process analog of
#      val_gpu_parity_script's TEXT-generation helpers in STADE.jl --
#      same shapes, real values instead of literal source) -----------

# mirrors _val_init_stacks_local in every stade_validate_gpu_file/
# stade_validate_jacc_file-emitted script: a Tier A/no-stack kernel's
# initstacks_* returns `nothing` (call it with zero stacks), a
# single-stack kernel returns a bare value (not a tuple), and every
# other kernel returns a tuple already in the right call order.
function gval_live_init_stacks(fn, extra_args)
    r = Base.invokelatest(fn, extra_args...)
    r === nothing && return ()
    r isa Tuple && return r
    return (r,)
end

# a stack_arg_name is either a scalar_int kernel argument (shared
# verbatim, unwrapped, between the CPU and device calls) or a values-
# Dict entry of its own -- and that entry might itself be a scalar or
# an array, so `wrap` is only applied when it actually is one. `wrap`
# is `identity` for the CPU side, CuArray/JACC.array for the device
# side.
function gval_stack_extra(names::Vector{Symbol}, int_args::Dict, values::Dict, wrap)
    return Any[haskey(int_args, n) ? int_args[n] :
               (values[n] isa AbstractArray ? wrap(deepcopy(values[n])) : values[n])
               for n in names]
end

# Builds one call's positional argument list AND a name-keyed dict of
# the actual (mutable) array objects passed in, so they can be
# re-inspected after the call completes without needing to reconstruct
# anything -- a Julia Array/CuArray/JACC.array is passed by reference,
# so whatever the callee wrote into it is already visible through the
# SAME object we still hold a handle to here. `wrap` is `identity` for
# the CPU call, CuArray/JACC.array for the device call -- everything
# else about the argument-ordering rule is backend-independent (see
# val_gpu_call_args's identical ordering in STADE.jl): scalar_int bare,
# scalar_float/array_float as a (value, shadow) pair, array_int as a
# bare value only (never differentiated).
function gval_build_call(sig, int_args::Dict, values::Dict, seed::Dict, wrap)
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
    return args, tracked
end

gval_relerr_scalar(a, b) = abs(a - b) / max(abs(a), abs(b), 1.0)
gval_relerr_arr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1.0))

# `dev_to_host` brings a device array back before comparing -- Array
# for CUDA, JACC.to_host for JACC. Compares EVERY mutated float array's
# shadow (the actual quantity of interest, the adjoint) and its value
# (mutated in place by the forward re-execution keep_push_pop=false
# checkpoints, so should also be bit-identical modulo the device's own
# floating-point reordering) -- and, for a read-only array_int table,
# its value too (should never differ at all; a cheap extra check).
function gval_compare(cpu_tracked::Dict, dev_tracked::Dict, dev_to_host)
    m = 0.0
    for (a, ct) in cpu_tracked
        dt = dev_tracked[a]
        m = max(m, gval_relerr_arr(ct.val, dev_to_host(dt.val)))
        ct.kind == :array_float && (m = max(m, gval_relerr_arr(ct.sh, dev_to_host(dt.sh))))
    end
    return m
end

# Runs `trials` reverse-mode seeds through both the CPU adjoint and one
# device plan (CUDA's or JACC's -- whichever (wrap, dev_to_host,
# atomic_macro) triple is passed in), eval'ing every kernel/host
# function into Main first. Returns the same (ok, max_rel_err) shape
# stade_validate_gpu_file/stade_validate_jacc_file's REMOTE verdict
# uses, just computed in-process instead of parsed from a pushed-back
# result.
function gval_check_backend(kernel, int_args::Dict, values::Dict, seeds::Vector,
                             adjoint_out, plan_init, plan_adj, stack_arg_names::Vector{Symbol},
                             wrap, dev_to_host, atomic_macro)
    for k in plan_init.kernels
        Core.eval(Main, k)
    end
    Core.eval(Main, plan_init.host)
    for k in plan_adj.kernels
        Core.eval(Main, k)
    end
    Core.eval(Main, plan_adj.host)

    cpu_init_fn = Core.eval(Main, adjoint_out.initstacks.args[1].args[1])
    cpu_adj_fn  = Core.eval(Main, adjoint_out.adjoint.args[1].args[1])
    dev_init_fn = Core.eval(Main, plan_init.host.args[1].args[1])
    dev_adj_fn  = Core.eval(Main, plan_adj.host.args[1].args[1])

    sig = kernel.sig
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    # same non-associative-atomic-summation reasoning as
    # stade_validate_gpu_file's resolved_rtol in STADE.jl -- scale the
    # default tolerance by how many @atomic/Atomix.@atomic sites the
    # generated adjoint actually has.
    atomic_sites = sum(val_count_atomic_sites(k, atomic_macro) for k in plan_adj.kernels; init = 0)
    rtol = 2.5e-14 * max(1, atomic_sites)

    max_err = 0.0
    for seed in seeds
        cpu_args, cpu_tracked = gval_build_call(sig, int_args, values, seed, identity)
        dev_args, dev_tracked = gval_build_call(sig, int_args, values, seed, wrap)
        stacks_cpu = gval_live_init_stacks(cpu_init_fn, gval_stack_extra(stack_arg_names, int_args, values, identity))
        stacks_dev = gval_live_init_stacks(dev_init_fn, gval_stack_extra(stack_arg_names, int_args, values, wrap))

        ret_cpu = Base.invokelatest(cpu_adj_fn, cpu_args..., stacks_cpu...)
        ret_dev = Base.invokelatest(dev_adj_fn, dev_args..., stacks_dev...)

        m = gval_compare(cpu_tracked, dev_tracked, dev_to_host)
        if !isempty(scalar_args)
            ret_cpu_t = length(scalar_args) == 1 ? (ret_cpu,) : ret_cpu
            ret_dev_t = length(scalar_args) == 1 ? (ret_dev,) : ret_dev
            for i in eachindex(scalar_args)
                m = max(m, gval_relerr_scalar(ret_cpu_t[i], ret_dev_t[i]))
            end
        end
        max_err = max(max_err, m)
    end
    return (ok = max_err <= rtol, max_rel_err = max_err, rtol = rtol)
end

# ---- top-level driver ------------------------------------------------

function validate_corpus_gpu(dir::String = "val-corpus"; trials::Int = 3,
                              fuse_ii_loops::Bool = false,
                              run_cuda::Bool = true, run_jacc::Bool = true,
                              write_generated::Bool = true,
                              only::Union{Vector{String},Nothing} = nothing)
    for f in readdir(dir)
        if endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl") ||
           endswith(f, ".yaml") || endswith(f, "_cuda.jl") || endswith(f, "_jacc.jl")
            rm(joinpath(dir, f))
        end
    end
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl") ||
                    endswith(f, "_cuda.jl") || endswith(f, "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))

    cuda_backend = cgen_backend_cuda()
    jacc_atomic_macro = Expr(:., :Atomix, QuoteNode(Symbol("@atomic")))

    results = NamedTuple[]
    for name in names
        path = joinpath(dir, name * ".jl")
        yp = io_default_yaml_path(path)
        isfile(yp) || stade_generate_baseline_file(path; yaml_path = yp)
        primal_expr = io_read_corpus_entry(path)
        kernel = parse_kernel(primal_expr)
        baseline = io_read_baseline_yaml(yp)
        val_coerce_int_arrays!(kernel, baseline.values)

        # ---- step 1: differentiate the monoprocessor primal ---------
        local adjoint_out
        try
            adjoint_out = stade_adjoint(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse_ii_loops)
        catch e
            println("  !! ", name, " [adjoint] ", first(split(sprint(showerror, e), "\n")))
            push!(results, (kernel = name, backend = :adjoint, status = :gen_error, max_rel_err = NaN))
            continue
        end
        if write_generated
            open(joinpath(dir, name * "_b.jl"), "w") do f
                write(f, io_expr_to_source(adjoint_out.initstacks))
                write(f, io_expr_to_source(adjoint_out.adjoint))
            end
        end
        stack_arg_names = Symbol[a for a in adjoint_out.initstacks.args[1].args[2:end]]
        seeds = [val_random_values_like(kernel, baseline.values) for _ in 1:trials]

        # ---- step 2 + 3: CUDA codegen, then execute ------------------
        if run_cuda
            try
                plan_init = stade_gpu(adjoint_out.initstacks, cuda_backend)
                plan_adj  = stade_gpu(adjoint_out.adjoint, cuda_backend)
                if write_generated
                    open(joinpath(dir, name * "_cuda.jl"), "w") do f
                        for k in plan_init.kernels
                            write(f, io_expr_to_source(k))
                        end
                        write(f, io_expr_to_source(plan_init.host))
                        for k in plan_adj.kernels
                            write(f, io_expr_to_source(k))
                        end
                        write(f, io_expr_to_source(plan_adj.host))
                    end
                end
                r = gval_check_backend(kernel, baseline.int_args, baseline.values, seeds,
                                        adjoint_out, plan_init, plan_adj, stack_arg_names,
                                        CuArray, Array, cuda_backend.atomic_macro)
                push!(results, (kernel = name, backend = :cuda, status = r.ok ? :ok : :FAIL, max_rel_err = r.max_rel_err))
            catch e
                println("  !! ", name, " [cuda] ", first(split(sprint(showerror, e), "\n")))
                push!(results, (kernel = name, backend = :cuda, status = :gen_error, max_rel_err = NaN))
            end
        end

        # ---- step 2 + 3: JACC codegen, then execute ------------------
        if run_jacc
            try
                plan_init = stade_jacc(adjoint_out.initstacks)
                plan_adj  = stade_jacc(adjoint_out.adjoint)
                if write_generated
                    open(joinpath(dir, name * "_jacc.jl"), "w") do f
                        for k in plan_init.kernels
                            write(f, io_expr_to_source(k))
                        end
                        write(f, io_expr_to_source(plan_init.host))
                        for k in plan_adj.kernels
                            write(f, io_expr_to_source(k))
                        end
                        write(f, io_expr_to_source(plan_adj.host))
                    end
                end
                r = gval_check_backend(kernel, baseline.int_args, baseline.values, seeds,
                                        adjoint_out, plan_init, plan_adj, stack_arg_names,
                                        x -> JACC.array(x), x -> JACC.to_host(x), jacc_atomic_macro)
                push!(results, (kernel = name, backend = :jacc, status = r.ok ? :ok : :FAIL, max_rel_err = r.max_rel_err))
            catch e
                println("  !! ", name, " [jacc] ", first(split(sprint(showerror, e), "\n")))
                push!(results, (kernel = name, backend = :jacc, status = :gen_error, max_rel_err = NaN))
            end
        end
    end

    for r in results
        println(rpad("$(r.kernel) [$(r.backend)]", 30), " ", r.status,
                isnan(r.max_rel_err) ? "" : "  max_rel_err=$(round(r.max_rel_err, sigdigits = 4))")
    end
    bad = count(r -> r.status != :ok, results)
    println("\n", length(results) - bad, "/", length(results), " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return results
end

validate_corpus_gpu()