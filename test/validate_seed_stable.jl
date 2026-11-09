# The seeds and draws the validators use must be the same on every Julia version.
#
#   julia validate_seed_stable.jl
#
# Fixed values (taken on Julia 1.10.11, checked on 1.11.9): the FNV seed of a few names; draws of Random.Xoshiro and of the default
# generator (floats, integer ranges, randperm, shuffle!); STADE.val_randn at the sizes where Julia's own randn differs between versions.
# A failure means that this Julia version draws other data than the one the values come from: the validators then test other data on
# this machine. Do not update a value without checking that the new Julia version is the cause.
# val_randn uses a logarithm and a cosine, so its values are compared to 1e-9, not bit for bit.
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random, SHA

fp(x::AbstractArray) = bytes2hex(sha256(reinterpret(UInt8, vec(collect(x)))))[1:12]
fp(x) = bytes2hex(sha256(string(x)))[1:12]
stat(A) = [A[1], A[end], sum(A), sum(abs2, A)]
checks = 0; bad = 0
function sk_check(label, got, want)
    global checks += 1
    if got == want
        println(rpad(label, 52), " ok")
    else
        global bad += 1
        println(rpad(label, 52), " FAIL  got ", got, "  expected ", want, "   (Julia ", VERSION, ")")
    end
end
function sk_close(label, got, want)
    global checks += 1
    if all(isapprox.(got, want; rtol = 1e-9))
        println(rpad(label, 52), " ok")
    else
        global bad += 1
        println(rpad(label, 52), " FAIL  got ", got, "  expected ", want, "   (Julia ", VERSION, ")")
    end
end

# seeds
sk_check("val_stable_seed(\"abc\")", STADE.val_stable_seed("abc"), 0xe71fa2190541574b)
sk_check("val_stable_seed(\"unet\")", STADE.val_stable_seed("unet"), 0x64c1dfe4857eb8bb)
sk_check("val_stable_seed(:unet) equals the string's", STADE.val_stable_seed(:unet), STADE.val_stable_seed("unet"))

# what the draws build on: floats, integer ranges, randperm and (our) shuffle! of Xoshiro and of the default generator
sk_check("Xoshiro(5) rand(3)", fp(rand(Random.Xoshiro(5), 3)), "28a825642733")
sk_check("Xoshiro(5) rand(10^6)", fp(rand(Random.Xoshiro(5), 1000000)), "dfa6c5319d98")
sk_check("Xoshiro(5) rand(1:97, 1000)", fp(rand(Random.Xoshiro(5), 1:97, 1000)), "a940fb4f7a94")
sk_check("Xoshiro(5) rand(1:1000, 10^5)", fp(rand(Random.Xoshiro(5), 1:1000, 100000)), "f71cc2efe69e")
sk_check("Xoshiro(5) rand(0:2, 50)", fp(rand(Random.Xoshiro(5), 0:2, 50)), "6afe65229f1f")
sk_check("Xoshiro(5) randperm(1000)", fp(randperm(Random.Xoshiro(5), 1000)), "11297ab5d81c")
sk_check("val_shuffle!(Xoshiro(5), 1:7)", fp(STADE.val_shuffle!(Random.Xoshiro(5), collect(1:7))), "8ff51c96cb42")
sk_check("val_shuffle!(Xoshiro(5), 1:10^5)", fp(STADE.val_shuffle!(Random.Xoshiro(5), collect(1:100000))), "0aef443eb1db")
Random.seed!(5)
sk_check("seed!(5) then rand(2:5), rand()", fp((rand(2:5), rand())), "59adb42d6656")

# val_randn: first, last, sum and sum of squares
sk_close("val_randn(Xoshiro(5), 3)", stat(STADE.val_randn(Random.Xoshiro(5), 3)), [2.416547687122585, -0.7958130054071307, 1.4808802847422582, 6.492580316065381])
sk_close("val_randn(Xoshiro(5), 1000)", stat(STADE.val_randn(Random.Xoshiro(5), 1000)), [2.416547687122585, 1.0908640603062327, -9.847881789408522, 1027.8518947088467])
sk_close("val_randn(Xoshiro(5), 10^5)", stat(STADE.val_randn(Random.Xoshiro(5), 100000)), [2.416547687122585, 0.38438758377031407, 126.25860343825198, 100189.9995648914])
Random.seed!(5)
sk_close("val_randn(300, 300), default generator", stat(STADE.val_randn(300, 300)), [2.416547687122585, 1.9893035916647346, 164.96631114621945, 90127.56921646207])
sk_check("val_randn(rng, n) equals n val_normal(rng) draws", STADE.val_randn(Random.Xoshiro(9), 50) == (r = Random.Xoshiro(9); [STADE.val_normal(r) for _ in 1:50]), true)
let x = STADE.val_randn(Random.Xoshiro(11), 1000000)
    sk_check("val_randn: 10^6 draws have mean 0, variance 1 (to 5e-3)", (abs(sum(x) / length(x)) < 5e-3, abs(sum(abs2, x) / length(x) - 1) < 5e-3), (true, true))
end

# The validators must not use what is not stable: MersenneTwister, a hash as a seed, Julia's own randn, or Julia's own shuffle!.
# This file names them in strings and comments, so it is the one file not scanned.
let
    files = [f for f in sort(readdir(@__DIR__)) if endswith(f, ".jl") && f != "validate_seed_stable.jl"]
    for (label, pat) in (("MersenneTwister", r"MersenneTwister"), ("hash(", r"(?<![\w.])hash\("), ("bare randn(", r"(?<![\w.])randn\("), ("shuffle!(", r"(?<!\w)shuffle!\("))
        hits = [f for f in files if occursin(pat, read(joinpath(@__DIR__, f), String))]
        sk_check("no $label in the $(length(files)) test files", hits, String[])
    end
end

println(bad == 0 ? "all $checks seed checks passed" : "$bad of $checks seed checks failed")
exit(bad == 0 ? 0 : 1)
