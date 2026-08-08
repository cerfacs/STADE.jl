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

validate_corpus()