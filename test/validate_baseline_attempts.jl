# ==================== validate_baseline_attempts.jl =======================
# Checks that val_generate_baseline redraws enough (v0.4.42). The transformer accepts about 6% of its integer draws (range 3 to 6); with 16 redraws,
# 21 of 60 seeds failed on Julia 1.10, and a Julia 1.11 run of the cluster chain failed ownership the same way. Which seeds fail at a given number of
# redraws depends on the Julia version (on 1.11 none of seeds 13, 14, 15, 16 and 18 fails at 16). The test therefore uses 2 redraws, which fail for
# about 88% of the draws on any version, and requires the default to succeed on every seed.
#
#   julia validate_baseline_attempts.jl

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

function validate_baseline_attempts()
    path = joinpath(@__DIR__, "val-corpus", "transformer.jl")
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal)
    ok = true
    nfew = 0
    # which seeds fail at a few redraws depends on the Julia version, so several seeds are tried and at least one must fail
    for s in (13, 14, 15, 16, 18)
        Random.seed!(s)
        rfew = try STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 6, attempts = 2) catch e e end
        failedfew = !(rfew isa NamedTuple)
        nfew += failedfew
        # the default must succeed on every seed
        Random.seed!(s)
        rd = try STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 6) catch e e end
        okd = rd isa NamedTuple
        println("seed ", s, ": 2 attempts ", failedfew ? "fail" : "ok", ", default ", okd ? "ok" : "FAIL")
        ok &= okd
    end
    # none failing at 2 would leave the test with nothing to show
    nfew > 0 || (println("no seed fails at 2 attempts: the test is stale"); ok = false)
    println(ok ? "PASS validate_baseline_attempts" : "FAIL validate_baseline_attempts")
    return ok
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_baseline_attempts() ? 0 : 1)
end
