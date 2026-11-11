# Index forms a scalar reduction may read (cgen_idiomatic_scalar_reduction): `x[i + c]`, a fixed index in the other dimension
# (`r[i, 1]`), and a flattened 2D nest (`x[div(i - 1, m) + a, mod(i - 1, m) + b]`). The CUDA text is a mapreduce over views and the
# JACC text a closure over the loop index; both are evaluated here on host arrays and compared with the loop itself. Refused forms must
# stay on the host. Zero-trip and one-iteration loops are included (rule 11 of skill-stade-dev).
using Random
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

function rf_loop(src::String)
    path = tempname() * ".jl"; write(path, src)
    e = try STADE.io_read_corpus_entry(path) finally rm(path; force = true) end
    k = STADE.parse_kernel(e)
    st = first(s for s in k.body if s.kind == :for)
    return (expr = e, kernel = k, loop = st)
end

function rf_gate(c)
    fn_args = Set{Symbol}(c.kernel.sig.args)
    return STADE.cgen_local_scalar_reductions(c.loop.body, c.loop.var, c.loop, c.kernel.body, fn_args)
end

# value of the CUDA lowering, evaluated on host arrays
function rf_eval_cuda(red, loop, args::Dict)
    (_, _, arrs, term) = red
    ex = STADE.cgen_idiomatic_reduction_value(arrs, term, loop.var, loop.lo, loop.step, loop.hi)
    f = Core.eval(Main, Expr(:->, Expr(:tuple, keys(args)...), ex))
    return Base.invokelatest(f, values(args)...)
end

# value of the JACC lowering: the closure is applied to every position, as @parallel_reduce does
function rf_eval_jacc(red, loop, args::Dict)
    (_, _, arrs, term) = red
    n = STADE.cgen_trip_count(loop.lo, loop.step, loop.hi)
    ex = STADE.jgen_idiomatic_reduction_value(arrs, term, loop.var, loop.lo, loop.step, n)
    ex.head == :let && (ex = ex.args[2].args[1])   # free host scalars are rebound by a `let`
    kw = ex.args[3]; call = ex.args[4]          # (range = n, closure(arrs...))
    clo = call.args[1]
    f = Core.eval(Main, Expr(:->, Expr(:tuple, keys(args)...),
        Expr(:block, Expr(:(=), :__n, kw.args[2]), Expr(:(=), :__f, clo),
             Expr(:call, :sum, Expr(:parameters, Expr(:kw, :init, 0.0)), Expr(:generator, Expr(:call, :__f, :__k, arrs...), Expr(:(=), :__k, Expr(:call, :(:), 1, :__n)))))))
    return Base.invokelatest(f, values(args)...)
end

function rf_reference(c, args::Dict, outname::Symbol)
    f = STADE.val_compile(c.expr)
    a = Any[deepcopy(args[s]) for s in c.kernel.sig.args]
    Base.invokelatest(f, a...)
    return a[findfirst(==(outname), c.kernel.sig.args)][1]
end

hdr(n) = "# $n\n# case\n"
# accepted cases: (name, source, args builder)
function rf_cases()
    rng = Random.Xoshiro(11)
    mk(dims...) = STADE.val_randn(rng, dims...)
    L = Any[]
    push!(L, ("plain a[i]*b[i]", hdr("a1") * "function a1(x, y, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + x[i] * y[i]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:x => mk(n + 4), :y => mk(n + 4), :out => zeros(1), :n => n)))
    push!(L, ("neighbours x[i-1]*x[i+1], 2:n-1", hdr("a2") * "function a2(x, out, n)\n    s = 0.0\n    for i = 2:n - 1\n        s = s + x[i - 1] * x[i + 1]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:x => mk(n + 4), :out => zeros(1), :n => n)))
    push!(L, ("offset by a symbol, x[i+k]", hdr("a3") * "function a3(x, y, out, n, k)\n    s = 0.0\n    for i = 1:n\n        s = s + x[i + k] * y[i]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:x => mk(n + 6), :y => mk(n + 4), :out => zeros(1), :n => n, :k => 2)))
    push!(L, ("strided loop with an offset", hdr("a4") * "function a4(x, out, n)\n    s = 0.0\n    for i = 2:3:n\n        s = s + x[i + 1] * x[i]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:x => mk(n + 4), :out => zeros(1), :n => n)))
    push!(L, ("column of a matrix, r[i, 1]", hdr("a5") * "function a5(r, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + r[i, 1] * r[i, 1]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:r => mk(n + 4, 3), :out => zeros(1), :n => n)))
    push!(L, ("row of a matrix, r[2, i+1]", hdr("a6") * "function a6(r, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + r[2, i + 1]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:r => mk(3, n + 4), :out => zeros(1), :n => n)))
    push!(L, ("flattened 2D, offset 2 (square)", hdr("a7") * "function a7(p, q, out, m)\n    s = 0.0\n    for i = 1:m * m\n        r = div(i - 1, m) + 2\n        c = mod(i - 1, m) + 2\n        s = s + p[r, c] * q[r, c]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:p => mk(n + 3, n + 3), :q => mk(n + 3, n + 3), :out => zeros(1), :m => n)))
    push!(L, ("flattened 2D, K x M with offset 1", hdr("a8") * "function a8(p, out, k, m)\n    s = 0.0\n    for i = 1:k * m\n        r = div(i - 1, m) + 1\n        c = mod(i - 1, m) + 1\n        s = s + p[r, c] * p[r, c]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:p => mk(n + 2, n + 5), :out => zeros(1), :k => n, :m => n + 2)))
    push!(L, ("flattened 2D, different offsets for two arrays", hdr("a9") * "function a9(p, q, out, k, m)\n    s = 0.0\n    for i = 1:k * m\n        r = div(i - 1, m) + 1\n        c = mod(i - 1, m) + 2\n        s = s + p[r, c] * q[r + 1, c]\n    end\n    out[1] = s\n    return nothing\nend\n",
        n -> Dict(:p => mk(n + 3, n + 6), :q => mk(n + 3, n + 6), :out => zeros(1), :k => n, :m => n + 2)))
    return L
end

# refused cases: the loop must stay on the host
function rf_refused()
    L = Any[]
    push!(L, ("stride in the index, x[2i]", hdr("b1") * "function b1(x, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + x[2 * i]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("diagonal x[i, i]", hdr("b2") * "function b2(x, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + x[i, i]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("a read through another array", hdr("b3") * "function b3(x, idx, out, n)\n    s = 0.0\n    for i = 1:n\n        j = idx[i]\n        s = s + x[j]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("accumulator in an index", hdr("b4") * "function b4(x, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + x[i + s]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("term reads the accumulator", hdr("b5") * "function b5(x, out, n)\n    s = 1.0\n    for i = 1:n\n        s = s + s * x[i]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("flattened 2D, loop starts at 2", hdr("b6") * "function b6(p, out, m)\n    s = 0.0\n    for i = 2:m * m\n        r = div(i - 1, m) + 1\n        c = mod(i - 1, m) + 1\n        s = s + p[r, c]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("flattened 2D, trip count is not K * M", hdr("b7") * "function b7(p, out, m, nn)\n    s = 0.0\n    for i = 1:nn\n        r = div(i - 1, m) + 1\n        c = mod(i - 1, m) + 1\n        s = s + p[r, c]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("flattened 2D, div and mod use different M", hdr("b8") * "function b8(p, out, m, k)\n    s = 0.0\n    for i = 1:m * m\n        r = div(i - 1, m) + 1\n        c = mod(i - 1, k) + 1\n        s = s + p[r, c]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("flattened 2D, transposed (mod first)", hdr("b9") * "function b9(p, out, m)\n    s = 0.0\n    for i = 1:m * m\n        r = div(i - 1, m) + 1\n        c = mod(i - 1, m) + 1\n        s = s + p[c, r]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("a 1D read and a decoded read together", hdr("b10") * "function b10(p, y, out, m)\n    s = 0.0\n    for i = 1:m * m\n        r = div(i - 1, m) + 1\n        c = mod(i - 1, m) + 1\n        s = s + p[r, c] * y[i]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("loop variable inside the offset, x[i + 2 * i]", hdr("b12") * "function b12(x, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + x[i + 2 * i]\n    end\n    out[1] = s\n    return nothing\nend\n"))
    push!(L, ("loop variable used as a value", hdr("b11") * "function b11(x, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + x[i] * i\n    end\n    out[1] = s\n    return nothing\nend\n"))
    return L
end

global fails = 0
function rf_report(ok, msg)
    global fails += ok ? 0 : 1
    println(ok ? "ok   " : "FAIL ", msg)
end

for (name, src, mkargs) in rf_cases()
    c = rf_loop(src)
    red = rf_gate(c)
    if red === nothing; rf_report(false, "$name: gate refused"); continue; end
    for n in (0, 1, 2, 5, 9)
        args = mkargs(n)
        ref = rf_reference(c, args, :out)
        for (be, f) in (("cuda", rf_eval_cuda), ("jacc", rf_eval_jacc))
            got = try f(red[1], c.loop, args) catch e; sprint(showerror, e) end
            ok = got isa Number && abs(got - ref) <= 1e-12 * max(abs(ref), 1.0)
            rf_report(ok, "$name  n=$n  $be" * (ok ? "" : "  got=$got want=$ref"))
        end
    end
end
for (name, src) in rf_refused()
    c = rf_loop(src)
    rf_report(rf_gate(c) === nothing, "refused: $name")
end

# the run-time threshold: a local scalar reduction switches to the device at 16 elements (a host loop reads the device arrays one element
# at a time); an array-element accumulator keeps 32768 (its alternative is an atomic kernel); reduction_threshold = 0 gives 0 for both
function rf_thresholds(src::String; kw...)
    path = tempname() * ".jl"; write(path, src)
    e = try STADE.io_read_corpus_entry(path) finally rm(path; force = true) end
    host = string(STADE.stade_gpu(e, STADE.cgen_backend_cuda(); kw...).host)
    jhost = string(STADE.stade_jacc(e; kw...).host)
    return (occursin("< 16", host), occursin("< 32768", host), occursin("< 0", host), occursin("< 16", jhost), occursin("< 32768", jhost), occursin("< 0", jhost))
end
local_src = hdr("t1") * "function t1(x, out, n)\n    s = 0.0\n    for i = 1:n\n        s = s + x[i] * x[i]\n    end\n    out[1] = s\n    return nothing\nend\n"
elem_src = hdr("t2") * "function t2(x, out, n)\n    for i = 1:n\n        out[1] = out[1] + x[i] * x[i]\n    end\n    return nothing\nend\n"
t = rf_thresholds(local_src);                          rf_report(t == (true, false, false, true, false, false), "local scalar reduction: default switches at 16 (cuda, jacc) $t")
t = rf_thresholds(local_src; reduction_threshold = 0); rf_report(t == (false, false, true, false, false, true), "local scalar reduction: threshold 0 stays 0 $t")
t = rf_thresholds(elem_src);                           rf_report(t[1] == false && t[2] == true && t[4] == false && t[5] == true, "array-element reduction keeps 32768 $t")
# captured host scalars are rebound by `let` (a reassigned capture is boxed and cannot compile on a device); no scalars, no `let`
e1 = STADE.jgen_idiomatic_reduction_value([:x], :(x[off + (r - 1) * m + i]), :i, 1, 1, :m)
rf_report(e1.head == :let && Set(a.args[1] for a in e1.args[1].args) == Set([:off, :r, :m]), "jacc closure: free scalars rebound by let")
e2 = STADE.jgen_idiomatic_reduction_value([:x, :y], :(x[i] * y[i]), :i, 1, 1, 7)
rf_report(e2.head == :macrocall, "jacc closure: no free scalars, no let (plain dotprod unchanged)")
println(fails == 0 ? "ALL OK" : "$fails FAILED")
exit(fails == 0 ? 0 : 1)
