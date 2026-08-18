include("STADE.jl")

# short, JSON-string-safe exception text for the `errmsg` result field --
# sprint(showerror, e) can run long (LoadError wraps the original with
# file/line, a MethodError lists candidates, etc.); truncate defensively
# since this rides inside one JSON string in the RunPod result payload.
gen_errmsg(e) = (s = sprint(showerror, e); length(s) > 2000 ? s[1:2000] * "...[truncated]" : s)

# ==================== per-backend descriptors ========================
const GPU_BACKEND_SPECS = Dict(
    :cuda => (suffix = "_cuda", genfile = stade_cuda_file),
    :jacc => (suffix = "_jacc", genfile = stade_jacc_file),
)

"""
    validate_corpus_gpu_gen_only(dir="val-corpus-gpu";
                         backends=(:cuda, :jacc), keep_push_pop=false,
                         site_level_tbr=true)

GPU-porting counterpart to `validate_corpus_keep_push_pop_false_site_level_tbr_true`.
Same corpus dir and generation flags by default -- `keep_push_pop=false`'s
`:indexed` snapshot strategy is precisely what makes every push/pop site
splittable onto a device kernel in the first place (see skill-stade.md).

Unlike the CPU/GPU numeric validators, this script does NOT compile,
`include`, or execute any generated code -- it only exercises the codegen
pipeline (`stade_tangent_file`/`stade_adjoint_file`/`stade_hvp_file`, then
each backend's `stade_cuda_file`/`stade_jacc_file` port) and reports
whether each (kernel, mode, backend) combination generated successfully.
This makes it safe to run without a working CUDA.jl/JACC.jl install or
any GPU hardware -- it only checks that codegen itself doesn't throw
(e.g. `cgen_device_assign`'s data-race refusal, or any other compile-time
failure in `cgen_`/`jgen_`'s own logic).

For every kernel and mode (tangent/adjoint/hvp):
  1. Generates the CPU `_d.jl`/`_b.jl`/`_hv.jl` file
     (same as the CPU validator).
  2. For every backend in `backends`, converts that file to the backend
     (`stade_cuda_file`/`stade_jacc_file`) and writes the result to disk.
Neither file is compiled or executed.

Status values: `:gen_ok` (codegen succeeded) or `:gen_error` (codegen --
CPU-side or the backend port -- threw).
"""
function validate_corpus_gpu_gen_only(dir::String = "val-corpus-gpu";
                              backends::Tuple = (:cuda, :jacc),
                              keep_push_pop::Bool = false, site_level_tbr::Bool = true)
    for f in readdir(dir)
        if endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl") ||
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

        # ---- CPU side: generate the reference _d.jl/_b.jl/_hv.jl file
        cpu_ok = Dict{Symbol,Bool}()
        for mode in mode_order
            gen_fn, suffix = generators[mode]
            cpu_path = joinpath(dir, name * suffix)
            try
                gen_fn(path, cpu_path; keep_push_pop = keep_push_pop, site_level_tbr = site_level_tbr)
                cpu_ok[mode] = true
                push!(results, (kernel = name, mode = mode, backend = :cpu, status = :gen_ok))
            catch e
                cpu_ok[mode] = false
                push!(results, (kernel = name, mode = mode, backend = :cpu, status = :gen_error, errmsg = gen_errmsg(e)))
            end
        end

        # ---- backend side: port each mode that CPU-generated OK
        for b in backends
            spec = GPU_BACKEND_SPECS[b]
            for mode in mode_order
                cpu_ok[mode] || continue
                _, suffix = generators[mode]
                cpu_path = joinpath(dir, name * suffix)
                gpu_path = joinpath(dir, name * suffix[1:end-3] * spec.suffix * ".jl")
                try
                    spec.genfile(cpu_path, gpu_path)
                    push!(results, (kernel = name, mode = mode, backend = b, status = :gen_ok))
                catch e
                    push!(results, (kernel = name, mode = mode, backend = b, status = :gen_error, errmsg = gen_errmsg(e)))
                end
            end
        end
    end

    for r in results
        println(rpad("$(r.kernel) [$(r.mode)/$(r.backend)]", 36), " ", r.status,
                hasproperty(r, :errmsg) ? "  -- $(first(r.errmsg, 200))" : "")
    end
    return results
end

validate_corpus_gpu_gen_only()