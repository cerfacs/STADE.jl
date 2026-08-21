include("STADE.jl")
using Random

"""
    validate_corpus(dir="val-corpus"; trials=8)

For every `.jl` kernel in `dir`: generates its tangent (`_d.jl`), adjoint
(`_b.jl`), and hvp (`_hv.jl`) files alongside it, then runs finite-difference
/ JVP / VJP validation on each against a random baseline.

Before doing so, removes any pre-existing `_b.jl` / `_d.jl` / `_hv.jl` /
`.yaml` files already in `dir` -- these are all derived/cached artifacts
(generated code and cached random baselines) from prior runs, and a stale
one lying around silently short-circuits regeneration (baseline `.yaml`
files are only (re)written when absent) and can validate a kernel against
a baseline or against generated code left over from a different STADE.jl
revision than the one currently loaded. Only kernel source `.jl` files are
the real inputs here, so everything else in `dir` is disposable.
"""
function validate_corpus(dir::String = "val-corpus"; trials::Int = 8)
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
                gen_fn(path, out_path; fuse_ii_loops = true)
            catch e
                push!(results, (kernel = name, mode = mode, status = :gen_error, max_rel_err = NaN))
                continue
            end

            fn = mode == :tangent ? stade_validate_tangent_file :
                 mode == :adjoint ? stade_validate_adjoint_file : stade_validate_hvp_file
            try
                # must match the flags the file was GENERATED with just
                # above -- the validator's own defaults would otherwise
                # regenerate and check unfused math regardless, so the
                # fused path would never actually be exercised here.
                r = fn(path; trials = trials, fuse_ii_loops = true)
                push!(results, (kernel = name, mode = mode, status = r.ok ? :ok : :FAIL, max_rel_err = r.max_rel_err))
            catch e
                push!(results, (kernel = name, mode = mode, status = :error, max_rel_err = NaN))
            end
        end

        # Exact tangent-vs-adjoint identity <Yb, J*Xd> == <J'*Yb, Xd>.
        # Generates nothing -- it cross-checks the two derivative codes
        # against each other rather than against the primal, so it needs
        # no file of its own. No epsilon and no truncation: it agrees to
        # ~1e-15 where the finite-difference oracles cap out around
        # 1e-8, which is what lets it see a systematic error the others
        # cannot. It does NOT validate the tangent (a bug shared by both
        # codes cancels in the identity), so it complements the three
        # checks above rather than replacing any of them.
        try
            r = stade_validate_dotprod_file(path; trials = trials, fuse_ii_loops = true)
            push!(results, (kernel = name, mode = :dotprod, status = r.ok ? :ok : :FAIL, max_rel_err = r.max_rel_err))
        catch e
            push!(results, (kernel = name, mode = :dotprod, status = :error, max_rel_err = NaN))
        end
    end
    for r in results
        println(rpad("$(r.kernel) [$(r.mode)]", 30), " ", r.status,
                isnan(r.max_rel_err) ? "" : "  max_rel_err=$(round(r.max_rel_err, sigdigits=4))")
    end
    bad = count(r -> r.status != :ok, results)
    println("\n", length(results) - bad, "/", length(results), " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return results
end

validate_corpus()