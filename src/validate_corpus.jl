include("STADE.jl")
using Random

"""
    validate_corpus(dir="val-corpus"; trials=8)

For every `.jl` kernel in `dir`: generates its tangent (`_d.jl`), adjoint
(`_b.jl`), and hvp (`_hv.jl`) files alongside it, then runs finite-difference
/ JVP / VJP validation on each against a random baseline.
"""
function validate_corpus(dir::String = "val-corpus"; trials::Int = 8)
    generators = Dict(:tangent => (stade_tangent_file, "_d.jl"),
                       :adjoint => (stade_adjoint_file, "_b.jl"),
                       :hvp     => (stade_hvp_file, "_hv.jl"))
    results = NamedTuple[]
    for f in sort(
        filter(
            f -> endswith(f, ".jl") && !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
            readdir(dir)
        )
    )
        name = splitext(f)[1]
        path = joinpath(dir, f)
        Random.seed!(hash(name))   # reproducible baseline per kernel across runs
        for mode in (:tangent, :adjoint, :hvp)
            gen_fn, suffix = generators[mode]
            out_path = joinpath(dir, name * suffix)
            try
                gen_fn(path, out_path)
            catch e
                push!(results, (kernel = name, mode = mode, status = :gen_error, max_rel_err = NaN))
                continue
            end

            fn = mode == :tangent ? stade_validate_tangent_file :
                 mode == :adjoint ? stade_validate_adjoint_file : stade_validate_hvp_file
            try
                r = fn(path; trials = trials)
                push!(results, (kernel = name, mode = mode, status = r.ok ? :ok : :FAIL, max_rel_err = r.max_rel_err))
            catch e
                push!(results, (kernel = name, mode = mode, status = :error, max_rel_err = NaN))
            end
        end
    end
    for r in results
        println(rpad("$(r.kernel) [$(r.mode)]", 30), " ", r.status,
                isnan(r.max_rel_err) ? "" : "  max_rel_err=$(round(r.max_rel_err, sigdigits=4))")
    end
    return results
end

# every GPU-backend-suffixed file this function itself writes, so a
# second run doesn't try to port its own previous output
gpu_port_suffixes() = ("_cuda.jl", "_amdgpu.jl", "_metal.jl", "_jacc.jl")

"""
    validate_gpu_ports(dir="val-corpus")

For every `.jl` file already in `dir` -- every original kernel AND every
`_d.jl`/`_b.jl`/`_hv.jl` file `validate_corpus` generated alongside it --
calls each of the four GPU-porting entry points (`stade_cuda_file`,
`stade_amdgpu_file`, `stade_metal_file`, `stade_jacc_file`) and checks
that the result is at least syntactically valid Julia.

This is a structural check only, not a correctness one: no GPU hardware
of any vendor is available in this environment to actually run anything
cgen_/jgen_ produce, so "parses" is the only oracle there is here (same
limitation noted throughout skill-stade.md's cgen_/jgen_ sections).
Run `validate_corpus` first so the `_d.jl`/`_b.jl`/`_hv.jl` files this
function also ports actually exist.

A multi-kernel corpus file (one with helper kernels called by a main
kernel, e.g. `advection_multi.jl`/`mg_vcycle_multi.jl` and everything
`validate_corpus` generates from them) is a known, documented gap, not
a bug: `stade_*_file`'s multi-kernel corpus support writes each
kernel's un-inlined primal copy with calls to its helpers still in it,
which cgen_/jgen_ don't yet recognize (see skill-stade.md). Those cases
are expected to report an :error status here, not :ok.
"""
function validate_gpu_ports(dir::String = "val-corpus")
    gpu_generators = Dict(:cuda => stade_cuda_file, :amdgpu => stade_amdgpu_file,
                           :metal => stade_metal_file, :jacc => stade_jacc_file)
    results = NamedTuple[]
    for f in sort(
        filter(
            f -> endswith(f, ".jl") && !any(endswith(f, s) for s in gpu_port_suffixes()),
            readdir(dir)
        )
    )
        name = splitext(f)[1]
        path = joinpath(dir, f)
        for backend in sort(collect(keys(gpu_generators)))
            gen_fn = gpu_generators[backend]
            out_path = joinpath(dir, name * "_" * string(backend) * ".jl")
            try
                gen_fn(path, out_path)
                Meta.parseall(read(out_path, String))
                push!(results, (file = f, backend = backend, status = :ok, detail = ""))
            catch e
                push!(results, (file = f, backend = backend, status = :error, detail = sprint(showerror, e)))
            end
        end
    end
    for r in results
        println(rpad("$(r.file) [$(r.backend)]", 34), " ", r.status)
    end
    n_ok = count(r -> r.status == :ok, results)
    println("\n$(n_ok)/$(length(results)) GPU ports parsed OK ",
            "(structural check only -- no GPU hardware available to run any of it; ",
            "failures on *_multi.jl-derived files are the documented multi-kernel-corpus gap, not a regression)")
    return results
end

validate_corpus()
validate_gpu_ports()