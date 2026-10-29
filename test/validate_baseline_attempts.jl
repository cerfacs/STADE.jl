# ==================== validate_baseline_attempts.jl =======================
# Checks that val_generate_baseline redraws enough (v0.4.42). The transformer accepts about 6% of its integer draws (range 3 to 6); with 16 redraws,
# 21 of 60 seeds failed on Julia 1.10, and a Julia 1.11 run of the cluster chain failed ownership the same way. Seeds 13, 14, 15, 16 and 18 are tried; the ones that fail at 16 depend on the Julia version.
#
#   julia validate_baseline_attempts.jl

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

function validate_baseline_attempts()
    path = joinpath(@__DIR__, "val-corpus", "transformer.jl")
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal)
    ok = true
    n16 = 0
    # which seeds fail at 16 redraws depends on the Julia version, so several seeds are tried and at least one must fail
    for s in (13, 14, 15, 16, 18)
        Random.seed!(s)
        r16 = try STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 6, attempts = 16) catch e e end
        failed16 = !(r16 isa NamedTuple)
        n16 += failed16
        # the default must succeed on every seed
        Random.seed!(s)
        rd = try STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 6) catch e e end
        okd = rd isa NamedTuple
        println("seed ", s, ": 16 attempts ", failed16 ? "fail" : "ok", ", default ", okd ? "ok" : "FAIL")
        ok &= okd
    end
    # none failing at 16 would leave the test with nothing to show
    n16 > 0 || (println("no seed fails at 16 attempts: the test is stale"); ok = false)
    println(ok ? "PASS validate_baseline_attempts" : "FAIL validate_baseline_attempts")
    return ok
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_baseline_attempts() ? 0 : 1)
end
