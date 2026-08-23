include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

"""
    validate_corpus(dir="val-corpus"; trials::Int = 8, keep_push_pop::Bool = false, fuse_ii_loops::Bool = true)

Run the tangent, adjoint, HVP, and dot-product oracles on every kernel in `dir`.

For each `.jl` kernel, this function generates a tangent file (`_d.jl`), an
adjoint file (`_b.jl`), and an HVP file (`_hv.jl`) next to it. It then checks
each generated file against a random baseline with finite differences (tangent,
adjoint, HVP) and, separately, against the exact identity
`<Yb, J*Xd> == <J'*Yb, Xd>` (dot-product mode).

Before it runs, the function deletes any `_b.jl`, `_d.jl`, `_hv.jl`, or `.yaml`
file already in `dir`. These files are derived output from an earlier run. A
stale copy can hide regeneration, since a baseline `.yaml` file is written only
when one is absent, or it can validate a kernel against generated code from a
different STADE revision than the one this script loads. Only the kernel
source `.jl` files are real input, so this script treats everything else in
`dir` as disposable.
"""
function validate_corpus(dir::String = "val-corpus"; trials::Int = 8, keep_push_pop::Bool = false, fuse_ii_loops::Bool = true)
    for f in readdir(dir)
        if endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl") || endswith(f, ".yaml")
            rm(joinpath(dir, f))
        end
    end
    generators = Dict(:tangent => (STADE.stade_tangent_file, "_d.jl"),
                       :adjoint => (STADE.stade_adjoint_file, "_b.jl"),
                       :hvp     => (STADE.stade_hvp_file, "_hv.jl"))
    results = NamedTuple[]
    for f in sort(
        filter(
            f -> endswith(f, ".jl") && (isempty(ARGS) || f in ARGS) && !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
            readdir(dir)
        )
    )
        name = splitext(f)[1]
        path = joinpath(dir, f)
        Random.seed!(hash(name))   # gives a reproducible baseline per kernel across runs
        for mode in (:tangent, :adjoint, :hvp)
            gen_fn, suffix = generators[mode]
            out_path = joinpath(dir, name * suffix)
            try
                gen_fn(path, out_path; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
            catch e
                push!(results, (kernel = name, mode = mode, status = :gen_error, max_rel_err = NaN))
                continue
            end

            fn = mode == :tangent ? STADE.stade_validate_tangent_file :
                 mode == :adjoint ? STADE.stade_validate_adjoint_file : STADE.stade_validate_hvp_file
            try
                # Pass the SAME flags used above for generation -- the validator's own
                # defaults would otherwise check a different mode's math than the one
                # under test, and the flagged path would go unexercised.
                r = fn(path; trials = trials, keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
                push!(results, (kernel = name, mode = mode, status = r.ok ? :ok : :FAIL, max_rel_err = r.max_rel_err))
            catch e
                println("  !! ", name, " [", mode, "] ", first(split(sprint(showerror, e), "\n"))[1:min(end,150)])
                push!(results, (kernel = name, mode = mode, status = :error, max_rel_err = NaN))
            end
        end

        # Exact tangent-vs-adjoint identity <Yb, J*Xd> == <J'*Yb, Xd>.
        # This mode generates no file -- it cross-checks the two derivative codes
        # against each other instead of against the primal, so it needs no file of
        # its own. It carries no epsilon and no truncation: it agrees to ~1e-15,
        # where the finite-difference oracles cap out around 1e-8, so it catches a
        # systematic error the others cannot. It does NOT validate the tangent (a
        # bug shared by both codes cancels in the identity), so it complements the
        # three checks above instead of replacing any of them.
        try
            r = STADE.stade_validate_dotprod_file(path; trials = trials, keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)
            push!(results, (kernel = name, mode = :dotprod, status = r.ok ? :ok : :FAIL, max_rel_err = r.max_rel_err))
        catch e
            println("  !! ", name, " [dotprod] ", first(split(sprint(showerror, e), "\n"))[1:min(end,150)])
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