include("STADE.jl")
using Random

"""
    validate_corpus_tapenade_adjoint(primal_dir="val-corpus",
                                      adjoint_dir="val-corpus-tapenade-adjoint";
                                      trials=8)

Validates a set of pre-generated (Tapenade) adjoint files against finite
differences of their corresponding primal kernels in `primal_dir`. For each
`<name>_b.jl` found in `adjoint_dir`, looks up the matching `<name>.jl` in
`primal_dir` and runs `stade_validate_adjoint_against_file`.
"""
function validate_corpus_tapenade_adjoint(primal_dir::String = "val-corpus",
                                           adjoint_dir::String = "val-corpus-tapenade-adjoint";
                                           trials::Int = 8)
    results = NamedTuple[]
    for f in sort(filter(f -> endswith(f, "_b.jl"), readdir(adjoint_dir)))
        name = f[1:end-5]  # strip "_b.jl"
        adjoint_path = joinpath(adjoint_dir, f)
        primal_path = joinpath(primal_dir, name * ".jl")

        if !isfile(primal_path)
            push!(results, (kernel = name, status = :no_primal, max_rel_err = NaN))
            continue
        end

        Random.seed!(hash(name))
        try
            r = stade_validate_adjoint_against_file(primal_path, adjoint_path; trials = trials)
            push!(results, (kernel = name, status = r.ok ? :ok : :FAIL, max_rel_err = r.max_rel_err))
        catch e
            push!(results, (kernel = name, status = :error, max_rel_err = NaN))
        end
    end
    for r in results
        println(rpad("$(r.kernel) [tapenade adjoint]", 34), " ", r.status,
                isnan(r.max_rel_err) ? "" : "  max_rel_err=$(round(r.max_rel_err, sigdigits=4))")
    end
    return results
end

validate_corpus_tapenade_adjoint()