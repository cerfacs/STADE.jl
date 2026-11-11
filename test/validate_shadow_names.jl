# A float variable named like the shadow of another float variable (primal name plus b, d or bd) must be refused at parse time.
# Before the guard, `w` and a local `wd` produced a tangent that assigned the local into the shadow array: wrong code, no error.
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

function sn_kernel(src::String)
    path = tempname() * ".jl"
    write(path, src)
    try
        return STADE.io_read_corpus_entry(path)
    finally
        rm(path; force = true)
    end
end

function sn_outcome(src::String)
    e = sn_kernel(src)
    try
        STADE.parse_kernel(e)
        return :accepted
    catch ex
        return occursin("shadow of", sprint(showerror, ex)) ? :refused : :other_error
    end
end

hdr(name) = "# $name\n# case\n"
cases = [
    ("local wd with array w",       :refused,  hdr("c1") * "function c1(w, y, n)\n    for i1 = 1:n\n        wd = w[i1] * 2.0\n        y[i1] = wd * wd\n    end\n    return nothing\nend\n"),
    ("local ub with array u",       :refused,  hdr("c2") * "function c2(u, y, n)\n    for i1 = 1:n\n        ub = u[i1] * 2.0\n        y[i1] = ub\n    end\n    return nothing\nend\n"),
    ("two arguments, x and xbd",    :refused,  hdr("c3") * "function c3(x, xbd, n)\n    for i1 = 1:n\n        xbd[i1] = x[i1]\n    end\n    return nothing\nend\n"),
    ("scalar s and local sd",       :refused,  hdr("c4") * "function c4(x, y, s, n)\n    for i1 = 1:n\n        sd = s * x[i1]\n        y[i1] = sd\n    end\n    return nothing\nend\n"),
    ("integer n and integer nd",    :accepted, hdr("c5") * "function c5(x, y, n)\n    nd = n - 1\n    for i1 = 1:nd\n        y[i1] = x[i1]\n    end\n    return nothing\nend\n"),
    ("unrelated float names",       :accepted, hdr("c6") * "function c6(x, y, n)\n    for i1 = 1:n\n        t1 = x[i1] * 2.0\n        y[i1] = t1\n    end\n    return nothing\nend\n"),
    ("float a and integer ad",      :accepted, hdr("c7") * "function c7(a, y, n)\n    ad = n\n    for i1 = 1:ad\n        y[i1] = a\n    end\n    return nothing\nend\n"),
]
global fails = 0
for (name, want, src) in cases
    got = sn_outcome(src)
    ok = got === want
    global fails += ok ? 0 : 1
    println(ok ? "ok   " : "FAIL ", rpad(name, 30), " want=", want, " got=", got)
end
# the refusal names both variables
msg = try STADE.parse_kernel(sn_kernel(cases[1][3])); "" catch ex; sprint(showerror, ex) end
nm_ok = occursin("`wd`", msg) && occursin("`w`", msg)
global fails += nm_ok ? 0 : 1
println(nm_ok ? "ok   " : "FAIL ", "message names both variables")
# the corpora stay accepted
for d in ("val-corpus", "battle-corpus", "int-state-corpus")
    dir = joinpath(@__DIR__, d); bad = String[]
    for f in sort(readdir(dir)); endswith(f, ".jl") || continue
        try STADE.parse_kernel(STADE.io_read_corpus_entry(joinpath(dir, f))) catch ex; occursin("shadow of", sprint(showerror, ex)) && push!(bad, f); end
    end
    global fails += isempty(bad) ? 0 : 1
    println(isempty(bad) ? "ok   " : "FAIL ", d, " has no kernel refused by the guard", isempty(bad) ? "" : ": " * join(bad, ","))
end
println(fails == 0 ? "ALL OK" : "$fails FAILED")
exit(fails == 0 ? 0 : 1)
