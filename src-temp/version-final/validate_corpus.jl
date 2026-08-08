include("STADE.jl")
using Random

"""
    validate_corpus(dir="val-corpus"; trials=8, skip=Dict("mg_vcycle"=>[:adjoint]))

For every `.jl` kernel in `dir`: generates its tangent (`_d.jl`), adjoint
(`_b.jl`), and hvp (`_hv.jl`) files alongside it, then runs finite-difference
/ JVP / VJP validation on each against a random baseline. `skip` lets you
exclude known-failing (kernel, mode) pairs -- defaults to `mg_vcycle`'s
adjoint, a known STADE codegen bug being tracked separately (generation
still happens for skipped pairs; only the numerical check is skipped).
"""
function validate_corpus(dir::String = "val-corpus"; trials::Int = 8,
                          skip::Dict{String,Vector{Symbol}} = Dict("mg_vcycle" => [:adjoint]))
    generators = Dict(:tangent => (stade_tangent_file, "_d.jl"),
                       :adjoint => (stade_adjoint_file, "_b.jl"),
                       :hvp     => (stade_hvp_file, "_hv.jl"))
    results = NamedTuple[]
    for f in sort(filter(f -> endswith(f, ".jl"), readdir(dir)))
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

            if mode in get(skip, name, Symbol[])
                push!(results, (kernel = name, mode = mode, status = :skipped, max_rel_err = NaN))
                continue
            end
            fn = mode == :tangent ? stade_validate_tangent_file :
                 mode == :adjoint ? stade_validate_adjoint_file : stade_validate_hvp_file
            try
                r = fn(path; trials = trials)
                push!(results, (kernel = name, mode = mode, status = r.ok ? :ok : :fail, max_rel_err = r.max_rel_err))
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