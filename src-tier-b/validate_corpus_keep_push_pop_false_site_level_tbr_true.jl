include("STADE.jl")
using Random

"""
    validate_corpus_keep_push_pop_false_site_level_tbr_true(dir="val-corpus-keep_push_pop_false-site_level_tbr_true"; trials=8)

Same as `validate_corpus`, but generates tangent/adjoint/hvp code with
`keep_push_pop=false` (the `:indexed` snapshot-storage strategy) instead
of the default `true` and `site_level_tbr=true` instead of the default `false`.

Before doing so, removes any pre-existing `_b.jl` / `_d.jl` / `_hv.jl` /
`.yaml` files already in `dir` -- these are all derived/cached artifacts
(generated code and cached random baselines) from prior runs, and a stale
one lying around silently short-circuits regeneration (baseline `.yaml`
files are only (re)written when absent) and can validate a kernel against
a baseline or against generated code left over from a different STADE.jl
revision than the one currently loaded. Only kernel source `.jl` files are
the real inputs here, so everything else in `dir` is disposable.
"""
function validate_corpus_keep_push_pop_false_site_level_tbr_true(dir::String = "val-corpus-keep_push_pop_false-site_level_tbr_true"; trials::Int = 8)
    for f in readdir(dir)
        if endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl") || endswith(f, ".yaml")
            rm(joinpath(dir, f))
        end
    end
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
                gen_fn(path, out_path; keep_push_pop = false, site_level_tbr = true)
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

validate_corpus_keep_push_pop_false_site_level_tbr_true()