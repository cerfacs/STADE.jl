include(joinpath(@__DIR__, "sabotage_run.jl"))
# validate_hfuse.jl
#
# Horizontal fusion (cgen_hfuse_*, gather_stencil level 5, the default): adjacent loops that do not depend on each other and cover abutting ranges become
# one loop with a branch per original range, guarded by a run-time test that every trip count is non-negative; the original loops stay in the other arm.
#   1. units: handmade loops, rewritten at level 4 and 5 and run on the host against the original for n from 0 to 12 (the other arm included);
#      each case says whether the fusion must fire
#   2. text checks: the guard, the branch order, one loop in the fused arm
#   3. sabotage: edits of the fusion, each of which a check must catch
using Random
if !isdefined(Main, :STADE)
    include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
end

# three loops over abutting ranges of v (the boundary points of a stencil and its interior); %A% %B% %C% are the three headers, %EXTRA% a statement after them
hf_src(; a = "i = 6:6", b = "i = n + 3:n + 3", c = "i = 7:n + 2", after = "", bodyc = "v[i] = u[i - 1] + 2.0 * u[i] + u[i + 1]", bodya = "v[i] = u[i] + u[i + 1]", bodyb = "v[i] = u[i - 1] - u[i]", pre = "") = """
function f(u, v, w, n)
    $pre
    for $a
        $bodya
    end
    for $b
        $bodyb
    end
    for $c
        $bodyc
    end
    $after
    return nothing
end
"""

# (label, source, the fusion must fire)
const HF_UNIT = [
    ("boundary points beside the interior, in source order 6, n+3, 7:n+2", hf_src(), true),
    ("the same loops in the order 7:n+2, 6, n+3", hf_src(a = "i = 7:n + 2", b = "i = 6:6", c = "i = n + 3:n + 3", bodya = "v[i] = u[i - 1] + 2.0 * u[i] + u[i + 1]", bodyb = "v[i] = u[i] + u[i + 1]", bodyc = "v[i] = u[i - 1] - u[i]"), true),
    ("two loops", """
function f(u, v, w, n)
    for i = 1:3
        v[i] = u[i] + 1.0
    end
    for i = 4:n
        v[i] = u[i] - 1.0
    end
    return nothing
end
""", true),
    ("different loop variables", """
function f(u, v, w, n)
    for i = 1:3
        v[i] = u[i] + 1.0
    end
    for k = 4:n
        v[k] = u[k] - 1.0
    end
    return nothing
end
""", true),
    ("three loops writing two arrays", """
function f(u, v, w, n)
    for i = 1:2
        v[i] = u[i] + 1.0
    end
    for i = 3:4
        w[i] = u[i] * 2.0
    end
    for i = 5:n
        v[i] = u[i] - u[1]
    end
    return nothing
end
""", true),
    ("a gap between the ranges", """
function f(u, v, w, n)
    for i = 1:3
        v[i] = u[i] + 1.0
    end
    for i = 5:n
        v[i] = u[i] - 1.0
    end
    return nothing
end
""", false),
    ("overlapping ranges", """
function f(u, v, w, n)
    for i = 1:3
        v[i] = u[i] + 1.0
    end
    for i = 3:n
        v[i] = u[i] - 1.0
    end
    return nothing
end
""", false),
    ("a loop reads what another writes", hf_src(bodyc = "v[i] = u[i - 1] + v[i + 1]"), false),
    ("a loop reads its own array at its own index", hf_src(bodyc = "v[i] = v[i] + u[i]"), true),
    ("a loop writes off its index", hf_src(bodyc = "v[i + 1] = u[i]"), false),
    ("a local scalar that is read after the loops", hf_src(bodyc = "t1 = u[i]\n        v[i] = t1", after = "w[1] = t1", pre = "t1 = 0.0"), false),
    ("a local scalar of the loop", hf_src(bodyc = "t1 = u[i - 1] + u[i + 1]\n        v[i] = t1 * 0.5", pre = "t1 = 0.0"), true),
    ("a loop with step 2", hf_src(c = "i = 7:2:n + 2"), false),
]

function hf_call(f, names, arrs, scal)
    cp = Dict(k => copy(v) for (k, v) in arrs)
    vals = Any[haskey(cp, nm) ? cp[nm] : scal[nm] for nm in names]
    Base.invokelatest(f, vals...)
    return cp
end

function hf_unit(src::String; ns = 0:12)
    expr = Meta.parse(src)
    gk = STADE.cgen_ingest(expr)
    g4 = STADE.cgen_gather_stencil(gk, 4)
    g5 = STADE.cgen_gather_stencil(gk, 5)
    # Julia 1.11 can crash when a collection runs between a fresh eval and its first call: keep the modules alive and the collector off here.
    m0, m5 = Module(), Module()
    GC.enable(false)
    f0 = Core.eval(m0, STADE.cgen_gather_function_expr(gk))
    f5 = Core.eval(m5, STADE.cgen_gather_function_expr(g5))
    names = [string(a) for a in gk.args]
    rng = MersenneTwister(hash(src))
    worst = 0.0
    for n in ns
        len = n + 8
        arrs = Dict("u" => randn(rng, len), "v" => randn(rng, len), "w" => randn(rng, len))
        r0, r5 = hf_call(f0, names, arrs, Dict{String,Any}("n" => n)), hf_call(f5, names, arrs, Dict{String,Any}("n" => n))
        for k in sort(collect(keys(r0)))
            worst = max(worst, maximum(abs.(r0[k] .- r5[k])))
        end
    end
    GC.enable(true)
    GC.gc()
    t4 = STADE.io_expr_to_source(STADE.cgen_gather_function_expr(g4))
    t5 = STADE.io_expr_to_source(STADE.cgen_gather_function_expr(g5))
    return (worst = worst, fired = t4 != t5, text4 = t4, text5 = t5, g5 = g5, mods = (m0, m5))
end

hf_count(s, pat) = length(collect(eachmatch(pat, s)))

function hf_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, src, want) in HF_UNIT
        try
            r = hf_unit(src)
            check(r.worst <= 1e-12 && r.fired == want, "$(label): fusion $(r.fired ? "fires" : "does not fire") (want $(want ? "fires" : "does not")), difference to the original $(r.worst)")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    r = hf_unit(hf_src())
    t = r.text5
    check(occursin(r"if min\(.*\) >= 0", t), "the fused arm is behind a test that every trip count is non-negative")
    check(hf_count(t, r"for i = ") == 4, "one fused loop and the three original loops (the other arm)")
    check(occursin("if i <= 6", t), "the first branch covers the lowest range")
    # level 4 is the v0.4.66 text
    plain = "function f(u, v, w, n)\n    for i = 2:n\n        v[i] = u[i]\n    end\n    return nothing\nend\n"
    pg = STADE.cgen_ingest(Meta.parse(plain))
    check(STADE.cgen_gather_stencil(pg, 4) == STADE.cgen_gather_stencil(pg, 5), "a kernel with one loop is the same at levels 4 and 5")
    return bad
end

function validate_hfuse(; sabotage::Bool = true, phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- horizontal fusion checks")
        bad += hf_checks()
    end
    if sabotage && :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        repo = abspath(joinpath(@__DIR__, ".."))
        text = read(joinpath(repo, "src", "STADE.jl"), String)
        cases = [
            ("the run-time trip test is dropped", "cond = Expr(:call, :>=, length(trips) == 1 ? trips[1] : Expr(:call, :min, trips...), 0)", "cond = Expr(:call, :>=, 1, 0)"),
            ("the run-time trip test uses max", "Expr(:call, :min, trips...)", "Expr(:call, :max, trips...)"),
            ("a gap between ranges is accepted", "hfuse_abuts(hi, lo) = (e = Expr(:call, :+, hi, 1); cgen_gather_ge(e, lo) && cgen_gather_ge(lo, e))", "hfuse_abuts(hi, lo) = (e = Expr(:call, :+, hi, 1); cgen_gather_ge(lo, e))"),
            ("an overlap is accepted", "hfuse_abuts(hi, lo) = (e = Expr(:call, :+, hi, 1); cgen_gather_ge(e, lo) && cgen_gather_ge(lo, e))", "hfuse_abuts(hi, lo) = (e = Expr(:call, :+, hi, 1); cgen_gather_ge(e, lo))"),
            ("a read of an array another loop writes is accepted", "            (arr in i.wr && length(idxs) == 1 && idxs[1] === L.var) || return false\n", ""),
            ("a read of an own array off its index is accepted", "(arr in i.wr && length(idxs) == 1 && idxs[1] === L.var) || return false", "arr in i.wr || return false"),
            ("a write off the loop index is accepted", "            at_own || return nothing\n", ""),
            ("a local scalar that is read later is fused", "        isempty(i.locals) || isempty(ii_escapes_nested(root, L.body, i.locals)) || return nothing\n    end\n    first_loop", "    end\n    first_loop"),
            ("any step is accepted", "(L.kind == :for && L.step isa Integer && L.step == 1) || return nothing\n    isempty(L.body)", "(L.kind == :for) || return nothing\n    isempty(L.body)"),
            ("the branch chain tests the wrong end", "cond = Expr(:call, :<=, var, Ls[order[k]].hi), then = arms[k], els = chain", "cond = Expr(:call, :<=, var, Ls[order[k]].lo), then = arms[k], els = chain"),
            ("the loops are not ordered by position", "    order = cgen_hfuse_order(Ls)\n    order === nothing && return nothing\n", "    order = collect(1:length(Ls))\n"),
            ("the fused loop ends at the first loop's end", "fused = merge(first_loop, (hi = last_loop.hi, body = chain))", "fused = merge(first_loop, (hi = first_loop.hi, body = chain))"),
            ("the original loops are dropped from the other arm", "then = NamedTuple[fused], els = NamedTuple[Ls...])", "then = NamedTuple[fused], els = NamedTuple[])"),
            ("fusion is tried at level 4", "        lv >= 5 && (g = merge(g, (body = cgen_hfuse_body", "        lv >= 4 && (g = merge(g, (body = cgen_hfuse_body"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases[intersect(eachindex(cases), slice)]
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    include($(repr(joinpath(@__DIR__, "validate_hfuse.jl"))))
                    n = hf_checks()
                    println("RESULT ", n)
                """
                out = sab_read(script)
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all horizontal fusion checks passed" : "*** $bad horizontal fusion check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_hfuse() == 0 ? 0 : 1)
end
