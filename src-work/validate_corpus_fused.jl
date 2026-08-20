include("STADE.jl")
using Random

# validate_corpus.jl generates with fuse_ii_loops=true but calls the validator
# with its default (false), so the fused code is never finite-difference tested.
# This variant threads the flag through, as the keep_push_pop fix already did.
function validate_corpus_fused(dir::String = "val-corpus"; trials::Int = 8, fuse::Bool = true)
    for f in readdir(dir)
        (endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")) && rm(joinpath(dir, f))
    end
    gens = Dict(:tangent => (stade_tangent_file, "_d.jl"),
                :adjoint => (stade_adjoint_file, "_b.jl"),
                :hvp     => (stade_hvp_file, "_hv.jl"))
    fails = 0
    for f in sort(filter(f -> endswith(f, ".jl") &&
                              !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl")),
                         readdir(dir)))
        name = splitext(f)[1]; path = joinpath(dir, f)
        Random.seed!(hash(name))
        for mode in (:tangent, :adjoint, :hvp)
            gen_fn, suffix = gens[mode]
            try
                gen_fn(path, joinpath(dir, name * suffix); fuse_ii_loops = fuse)
            catch e
                println(rpad("$name [$mode]", 30), " gen_error"); fails += 1; continue
            end
            fn = mode == :tangent ? stade_validate_tangent_file :
                 mode == :adjoint ? stade_validate_adjoint_file : stade_validate_hvp_file
            try
                r = fn(path; trials = trials, fuse_ii_loops = fuse)
                r.ok || (fails += 1)
                println(rpad("$name [$mode]", 30), " ", r.ok ? "ok" : "FAIL",
                        "  max_rel_err=", round(r.max_rel_err, sigdigits = 4))
            catch e
                println(rpad("$name [$mode]", 30), " error"); fails += 1
            end
        end
    end
    println("\nfuse=", fuse, "  failures: ", fails)
    return fails
end

validate_corpus_fused()