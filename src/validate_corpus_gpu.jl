include("STADE.jl")
using Random

# ==================== gpu_* (device-array call-arg builders) ========
# Mirrors val_tangent_call_args/val_adjoint_call_args/val_hvp_call_args
# exactly (skill-stade purity rule: duplicate, don't reach into another
# stage's private convention), except every float/int ARRAY argument is
# converted to the backend's device array type before the call, and
# every array-typed OUTPUT is converted back to a host `Array` before
# being hostified for `val_flatten`. Scalar args/outputs are passed
# through untouched -- cgen_/jgen_'s host wrapper keeps scalar
# loop-bound/return-scalar arithmetic on the host regardless of backend
# (see cgen_emit/jgen_emit: `gk.args` are the SAME positional args, in
# the SAME order, as the CPU-generated function this was ported from).

gpu_arrify(to_device, v) = v isa AbstractArray ? Base.invokelatest(to_device, v) : v
# same world-age reason gpu_arrify above wraps its call in
# Base.invokelatest: `Array(::CuArray)` internally dispatches through
# `size(::CuArray)`, a method CUDACore only defines once `using CUDA`
# actually runs -- which happens dynamically, inside a generated
# _cuda.jl/_jacc.jl file's own preamble, strictly AFTER this function
# was first compiled. Without invokelatest here, every GPU-side result
# readback (gpu_hostify wraps every array-typed tangent/adjoint/hvp
# output before comparison) hits "method too new to be called from
# this world context" -- confirmed against a live run: this fired
# uniformly across every kernel/mode, even ones whose actual on-device
# math ran fine, because the crash was in the comparison step, not the
# kernel itself.
gpu_hostify(v) = v isa AbstractArray ? Base.invokelatest(Array, v) : v

# short, JSON-string-safe exception text for the `errmsg` result field --
# sprint(showerror, e) can run long (LoadError wraps the original with
# file/line, a MethodError lists candidates, etc.); truncate defensively
# since this rides inside one JSON string in the RunPod result payload.
gpu_errmsg(e) = (s = sprint(showerror, e); length(s) > 2000 ? s[1:2000] * "...[truncated]" : s)

function gpu_tangent_call_args(sig, int_args::Dict, values::Dict, dvalues::Dict, to_device)
    call = Any[]
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(call, int_args[a])
        elseif k in (:scalar_float, :array_float)
            push!(call, gpu_arrify(to_device, deepcopy(values[a])))
            push!(call, gpu_arrify(to_device, deepcopy(dvalues[a])))
        else
            push!(call, gpu_arrify(to_device, deepcopy(values[a])))
        end
    end
    return call
end

function gpu_adjoint_call_args(sig, int_args::Dict, values::Dict, seed::Dict, stacks, to_device)
    call = Any[]
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(call, int_args[a])
        elseif k in (:scalar_float, :array_float)
            push!(call, gpu_arrify(to_device, deepcopy(values[a])))
            push!(call, gpu_arrify(to_device, deepcopy(seed[a])))
        else
            push!(call, gpu_arrify(to_device, deepcopy(values[a])))
        end
    end
    # `stacks` is already device-resident: under keep_push_pop=false the
    # backend's own initstacks_*_cuda/_jacc allocates directly on-device
    # (cgen_stack_device_expr / jgen_stack_device_expr) from the same
    # scalar free_vars every backend shares -- nothing to convert here.
    append!(call, stacks)
    return call
end

function gpu_hvp_call_args(sig, int_args::Dict, values::Dict, seed::Dict, dvalues::Dict, dseed::Dict, stacks, to_device)
    call = Any[]
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(call, int_args[a])
        elseif k in (:scalar_float, :array_float)
            push!(call, gpu_arrify(to_device, deepcopy(values[a])))
            push!(call, gpu_arrify(to_device, deepcopy(seed[a])))
        else
            push!(call, gpu_arrify(to_device, deepcopy(values[a])))
        end
    end
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        push!(call, gpu_arrify(to_device, deepcopy(dvalues[a])))
        push!(call, gpu_arrify(to_device, deepcopy(dseed[a])))
    end
    append!(call, stacks)
    return call
end

# ---- call + extract-output helpers, one per generated file kind ----
# Same three-function shape as val_call_tangent/val_call_adjoint/
# val_call_hv, with gpu_hostify wrapping every array-typed output
# before it's handed to val_flatten (which itself iterates host
# memory -- e.g. via `vec`/`append!` -- and would trip
# `CUDA.allowscalar(false)`/JACC's own no-scalar-indexing rule if
# handed a still-device-resident array).

function gpu_call_tangent(tangent_fn, kernel, int_args::Dict, values::Dict, dvalues::Dict, to_device)
    sig = kernel.sig
    call_args = gpu_tangent_call_args(sig, int_args, values, dvalues, to_device)
    ret = Base.invokelatest(tangent_fn, call_args...)
    reassigned = val_reassigned_scalar_float_args(kernel)
    ret_tuple = isempty(reassigned) ? () : (length(reassigned) == 1 ? (ret,) : ret)
    ret_of = Dict(zip(reassigned, ret_tuple))
    out = Dict{Symbol,Any}()
    pos = 1
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            pos += 1
        elseif k in (:scalar_float, :array_float)
            shadow = call_args[pos + 1]
            out[a] = k == :scalar_float ? get(ret_of, a, dvalues[a]) : gpu_hostify(shadow)
            pos += 2
        else
            pos += 1
        end
    end
    return val_flatten(kernel, out)
end

function gpu_call_adjoint(adjoint_fn, initstacks_fn, kernel, int_args::Dict, values::Dict, seed::Dict, to_device;
                           stack_arg_names::Vector{Symbol} = Symbol[])
    sig = kernel.sig
    stack_extra_args = [haskey(int_args, n) ? int_args[n] : deepcopy(values[n]) for n in stack_arg_names]
    stacks = val_init_stacks(initstacks_fn, stack_extra_args)
    call_args = gpu_adjoint_call_args(sig, int_args, values, seed, stacks, to_device)
    ret = Base.invokelatest(adjoint_fn, call_args...)
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    ret_tuple = isempty(scalar_args) ? () : (length(scalar_args) == 1 ? (ret,) : ret)
    ret_of = Dict(zip(scalar_args, ret_tuple))
    out = Dict{Symbol,Any}()
    pos = 1
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            pos += 1
        elseif k in (:scalar_float, :array_float)
            shadow = call_args[pos + 1]
            out[a] = k == :scalar_float ? ret_of[a] : gpu_hostify(shadow)
            pos += 2
        else
            pos += 1
        end
    end
    return val_flatten(kernel, out)
end

function gpu_call_hv(hv_fn, initstacks_fn, kernel, int_args::Dict, values::Dict, seed::Dict,
                      dvalues::Dict, dseed::Dict, to_device; stack_arg_names::Vector{Symbol} = Symbol[])
    sig = kernel.sig
    stack_extra_args = [haskey(int_args, n) ? int_args[n] : deepcopy(values[n]) for n in stack_arg_names]
    stacks = val_init_stacks(initstacks_fn, stack_extra_args)
    call_args = gpu_hvp_call_args(sig, int_args, values, seed, dvalues, dseed, stacks, to_device)
    ret = Base.invokelatest(hv_fn, call_args...)
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    ret_tuple = isempty(scalar_args) ? () : ret
    hv_of = Dict{Symbol,Any}(a => ret_tuple[2i] for (i, a) in enumerate(scalar_args))
    n_lead = count(a -> sig.kinds[a] == :scalar_int, sig.args) +
             count(a -> sig.kinds[a] == :array_int, sig.args) +
             2 * count(a -> sig.kinds[a] in (:scalar_float, :array_float), sig.args)
    out = Dict{Symbol,Any}()
    i = 0
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        xbd_pos = n_lead + 2 * i + 2
        out[a] = sig.kinds[a] == :scalar_float ? hv_of[a] : gpu_hostify(call_args[xbd_pos])
        i += 1
    end
    return val_flatten(kernel, out)
end

# name of the def a raw Expr defines -- e.g. cpu_defs[1] for a `_b.jl`
# bundle is always `initstacks_foo_b`, matching io_write_kernel_corpus_file's
# documented `[generated_parts..., primal]` order (see stade_adjoint_file/
# stade_hvp_file: `Expr[generated.initstacks, generated.adjoint]` /
# `Expr[generated.initstacks, generated.hvp]`).
gpu_def_name(e::Expr) = e.args[1].args[1]

# the GPU-ported host function that corresponds to a given CPU def,
# looked up by name (cgen_host_fname/jgen_host_fname's own convention:
# original name + backend suffix) after `include`ing the backend's
# generated file into Main.
gpu_get_fn(cpu_def::Expr, suffix::String) = getfield(Main, Symbol(string(gpu_def_name(cpu_def)) * suffix))

# ==================== per-backend descriptors ========================
# `to_device` is resolved lazily (a plain closure over `Main.CUDA`/
# `Main.JACC`, evaluated only once the corresponding generated file's
# own preamble has `using`/`import`ed it) -- gpu_arrify calls it through
# `Base.invokelatest` since the backend module is only defined at
# runtime, strictly after this script itself was parsed/compiled.
const GPU_BACKEND_SPECS = Dict(
    :cuda => (suffix = "_cuda", genfile = stade_cuda_file, to_device = x -> Main.CUDA.CuArray(x)),
    :jacc => (suffix = "_jacc", genfile = stade_jacc_file, to_device = x -> Main.JACC.array(x)),
)

"""
    validate_corpus_gpu(dir="val-corpus-gpu";
                         backends=(:cuda, :jacc), keep_push_pop=false,
                         site_level_tbr=true, rtol=1e-6)

GPU-porting counterpart to `validate_corpus_keep_push_pop_false_site_level_tbr_true`.
Same corpus dir and generation flags by default -- `keep_push_pop=false`'s
`:indexed` snapshot strategy is precisely what makes every push/pop site
splittable onto a device kernel in the first place (see skill-stade.md) --
but instead of a central-difference oracle, this checks each backend's
ACTUAL EXECUTED output (via CUDA.jl / JACC.jl, run on whatever device(s)
this Julia process has) against the CPU-generated code's output on the
exact same random inputs. That's a stronger, cheaper check than
re-deriving a finite-difference oracle on-device: correctness of the
underlying math is already established by the CPU validator, so this only
needs to confirm the GPU port computes the SAME numbers. Run this only on
a machine with a working CUDA.jl and/or JACC.jl install -- the skill file
is explicit that no GPU hardware, of any vendor, had exercised `cgen_`/
`jgen_`'s output at any point before this script existed.

For every kernel and mode (tangent/adjoint/hvp), and every backend in
`backends`:
  1. (Re)generates and compiles the CPU `_d.jl`/`_b.jl`/`_hv.jl` file
     (same as the CPU validator) as the trusted reference.
  2. Converts that file to the backend (`stade_cuda_file`/`stade_jacc_file`),
     `include`s the result (this also `Pkg.add`s and `using`/`import`s the
     backend package the first time it's needed), and calls the generated
     host function with every float/int array argument converted to that
     backend's device array type via `to_device`.
  3. Flattens both outputs the same way the CPU validator does
     (`val_flatten`) and reports the max relative error between them.
  hvp reuses the SAME adjoint initstacks (both CPU and GPU) that mode
  :adjoint already validated, exactly mirroring `val_validate_hvp`'s own
  choice (see its own comment) rather than the hvp file's structurally-
  identical-but-separately-named `initstacks_*_hv`.

Status values: `:ok`/`:FAIL` (both sides ran; numbers agree/disagree),
`:gen_error` (codegen, or loading the generated file, failed -- e.g.
`cgen_device_assign`'s data-race refusal, or a compile error in the
generated source), `:run_error` (codegen succeeded but calling the
generated host function on-device threw -- e.g. a sequential host-side
loop scalar-indexing into a device array under `allowscalar(false)`; a
real, previously-unexercised failure mode this script exists to surface,
not something to silently work around here).
"""
function validate_corpus_gpu(dir::String = "val-corpus-gpu";
                              backends::Tuple = (:cuda, :jacc),
                              keep_push_pop::Bool = false, site_level_tbr::Bool = true,
                              rtol::Float64 = 1e-6)
    for f in readdir(dir)
        if endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl") || endswith(f, ".yaml") ||
           occursin("_cuda.jl", f) || occursin("_jacc.jl", f)
            rm(joinpath(dir, f))
        end
    end

    generators = Dict(:tangent => (stade_tangent_file, "_d.jl"),
                       :adjoint => (stade_adjoint_file, "_b.jl"),
                       :hvp     => (stade_hvp_file, "_hv.jl"))
    mode_order = (:tangent, :adjoint, :hvp)

    results = NamedTuple[]
    kernel_files = sort(filter(
        f -> endswith(f, ".jl") && !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
        readdir(dir)
    ))

    for f in kernel_files
        name = splitext(f)[1]
        path = joinpath(dir, f)
        primal_expr = io_read_corpus_entry(path)
        kernel = parse_kernel(primal_expr)
        Random.seed!(hash(name))   # reproducible per kernel, matching the CPU validator's convention
        yp = io_default_yaml_path(path)
        io_path_exists(yp) || stade_generate_baseline_file(path; yaml_path = yp)
        baseline = io_read_baseline_yaml(yp)
        val_coerce_int_arrays!(kernel, baseline.values)
        int_args = baseline.int_args
        values = baseline.values

        # ---- CPU side: generate + compile once per kernel, shared across backends
        cpu_defs = Dict{Symbol,Vector{Expr}}()
        cpu_ok = Dict{Symbol,Bool}()
        for mode in mode_order
            gen_fn, suffix = generators[mode]
            cpu_path = joinpath(dir, name * suffix)
            try
                gen_fn(path, cpu_path; keep_push_pop = keep_push_pop, site_level_tbr = site_level_tbr)
                cpu_defs[mode] = io_read_kernel_bundle(cpu_path)
                cpu_ok[mode] = true
            catch e
                cpu_ok[mode] = false
                msg = gpu_errmsg(e)
                for b in backends
                    push!(results, (kernel = name, mode = mode, backend = b, status = :gen_error, max_rel_err = NaN, errmsg = msg))
                end
            end
        end

        for b in backends
            spec = GPU_BACKEND_SPECS[b]

            # ---- generate + include this backend's port of each mode that CPU-generated OK
            gpu_ok = Dict{Symbol,Bool}()
            for mode in mode_order
                cpu_ok[mode] || continue
                _, suffix = generators[mode]
                cpu_path = joinpath(dir, name * suffix)
                gpu_path = joinpath(dir, name * suffix[1:end-3] * spec.suffix * ".jl")
                try
                    spec.genfile(cpu_path, gpu_path)
                    include(gpu_path)
                    gpu_ok[mode] = true
                catch e
                    gpu_ok[mode] = false
                    push!(results, (kernel = name, mode = mode, backend = b, status = :gen_error, max_rel_err = NaN, errmsg = gpu_errmsg(e)))
                end
            end

            # ---- :tangent
            if get(gpu_ok, :tangent, false)
                try
                    cpu_fn = val_compile(cpu_defs[:tangent][1])
                    gpu_fn = gpu_get_fn(cpu_defs[:tangent][1], spec.suffix)
                    dvalues = val_random_values_like(kernel, values)
                    cpu_out = val_call_tangent(cpu_fn, kernel, int_args, deepcopy(values), deepcopy(dvalues))
                    gpu_out = gpu_call_tangent(gpu_fn, kernel, int_args, deepcopy(values), deepcopy(dvalues), spec.to_device)
                    denom = max(maximum(abs.(cpu_out)), maximum(abs.(gpu_out)), 1e-12)
                    err = maximum(abs.(cpu_out .- gpu_out)) / denom
                    push!(results, (kernel = name, mode = :tangent, backend = b, status = err <= rtol ? :ok : :FAIL, max_rel_err = err))
                catch e
                    push!(results, (kernel = name, mode = :tangent, backend = b, status = :run_error, max_rel_err = NaN, errmsg = gpu_errmsg(e)))
                end
            end

            # ---- :adjoint (also stash CPU/GPU initstacks fn + stack_arg_names for :hvp reuse)
            adjoint_initstacks_cpu = nothing
            adjoint_initstacks_gpu = nothing
            stack_arg_names = Symbol[]
            if get(gpu_ok, :adjoint, false)
                try
                    initstacks_cpu = val_compile(cpu_defs[:adjoint][1])
                    adjoint_cpu = val_compile(cpu_defs[:adjoint][2])
                    initstacks_gpu = gpu_get_fn(cpu_defs[:adjoint][1], spec.suffix)
                    adjoint_gpu = gpu_get_fn(cpu_defs[:adjoint][2], spec.suffix)
                    stack_arg_names = val_def_arg_names(cpu_defs[:adjoint][1])
                    seed = val_random_values_like(kernel, values)
                    cpu_out = val_call_adjoint(adjoint_cpu, initstacks_cpu, kernel, int_args, deepcopy(values), seed;
                                                stack_arg_names = stack_arg_names)
                    gpu_out = gpu_call_adjoint(adjoint_gpu, initstacks_gpu, kernel, int_args, deepcopy(values), seed, spec.to_device;
                                                stack_arg_names = stack_arg_names)
                    denom = max(maximum(abs.(cpu_out)), maximum(abs.(gpu_out)), 1e-12)
                    err = maximum(abs.(cpu_out .- gpu_out)) / denom
                    push!(results, (kernel = name, mode = :adjoint, backend = b, status = err <= rtol ? :ok : :FAIL, max_rel_err = err))
                    adjoint_initstacks_cpu = initstacks_cpu
                    adjoint_initstacks_gpu = initstacks_gpu
                catch e
                    push!(results, (kernel = name, mode = :adjoint, backend = b, status = :run_error, max_rel_err = NaN, errmsg = gpu_errmsg(e)))
                end
            end

            # ---- :hvp (reuses adjoint's initstacks, both sides -- see docstring)
            if get(gpu_ok, :hvp, false)
                if adjoint_initstacks_cpu === nothing
                    push!(results, (kernel = name, mode = :hvp, backend = b, status = :run_error, max_rel_err = NaN, errmsg = "adjoint mode didn't produce usable initstacks (see its own :adjoint result)"))
                else
                    try
                        hv_cpu = val_compile(cpu_defs[:hvp][2])
                        hv_gpu = gpu_get_fn(cpu_defs[:hvp][2], spec.suffix)
                        seed = val_random_values_like(kernel, values)
                        dvalues = val_random_values_like(kernel, values)
                        dseed = val_zeros_like(kernel, values)
                        cpu_out = val_call_hv(hv_cpu, adjoint_initstacks_cpu, kernel, int_args, deepcopy(values), seed, dvalues, dseed;
                                               stack_arg_names = stack_arg_names)
                        gpu_out = gpu_call_hv(hv_gpu, adjoint_initstacks_gpu, kernel, int_args, deepcopy(values), seed, dvalues, dseed, spec.to_device;
                                               stack_arg_names = stack_arg_names)
                        denom = max(maximum(abs.(cpu_out)), maximum(abs.(gpu_out)), 1e-12)
                        err = maximum(abs.(cpu_out .- gpu_out)) / denom
                        push!(results, (kernel = name, mode = :hvp, backend = b, status = err <= rtol ? :ok : :FAIL, max_rel_err = err))
                    catch e
                        push!(results, (kernel = name, mode = :hvp, backend = b, status = :run_error, max_rel_err = NaN, errmsg = gpu_errmsg(e)))
                    end
                end
            end
        end
    end

    for r in results
        println(rpad("$(r.kernel) [$(r.mode)/$(r.backend)]", 36), " ", r.status,
                isnan(r.max_rel_err) ? "" : "  max_rel_err=$(round(r.max_rel_err, sigdigits=4))",
                hasproperty(r, :errmsg) ? "  -- $(first(r.errmsg, 200))" : "")
    end
    return results
end

validate_corpus_gpu()