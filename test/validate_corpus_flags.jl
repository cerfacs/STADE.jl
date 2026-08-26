# Runs validate_corpus over all four (keep_push_pop, fuse_ii_loops) combinations.
#
# validate_corpus.jl calls validate_corpus() at its own top level, so including
# it already performs the keep_push_pop=false, fuse_ii_loops=true run (the
# script's defaults). The loop below covers the remaining three rather than
# repeating that one.
#
# Each run regenerates every _b.jl/_d.jl/_hv.jl and deletes the .yaml baselines
# first, so the runs are independent and order does not matter.
#
#   cd test && julia validate_corpus_flags.jl

println("\n================ keep_push_pop=false  fuse_ii_loops=true ================")
include(joinpath(@__DIR__, "validate_corpus.jl"))

for (kpp, fuse) in ((false, false), (true, false), (true, true))
    println("\n================ keep_push_pop=", kpp, "  fuse_ii_loops=", fuse, " ================")
    validate_corpus(joinpath(@__DIR__, "val-corpus"); keep_push_pop = kpp, fuse_ii_loops = fuse)
end
