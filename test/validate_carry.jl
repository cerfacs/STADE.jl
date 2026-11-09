include(joinpath(@__DIR__, "sabotage_run.jl"))
# validate_carry.jl
#
# The carry fold (cgen_gather_carry): the zeroing loop that ends a sequential loop is folded into the gather loop that starts the next iteration,
# behind a flag that is 0 in the first iteration, and runs once after the loop if the loop ran at all (gather_stencil level 4, the default).
#   1. units: handmade time loops, rewritten at level 3 and 4 and run on the host against the original (n and nsteps small, zero-trip loops included);
#      each case says whether the fold must fire
#   2. text checks: the flag, the guard of the folded statements, the guard of the closing loop
#   3. sabotage: edits of cgen_gather_carry, each of which a check must catch
using Random
if !isdefined(Main, :STADE)
    include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
end

# the reverse sweep of heat1d_multi_b; %BODY% marks the place of an extra statement, %LOOP% the header of the time loop, %ZERO% the last loop
const CAR_BASE = """
function f(ub, vb, n, nsteps, c, m)
    %PRE%
    for t = %LOOP%
        for i = n - 1:-1:2
            __oldb_0 = ub[i]
            ub[i] = 0.0
            vb[i] = vb[i] + __oldb_0
            __cse_2 = 0.25__oldb_0
            vb[i - 1] = vb[i - 1] + __cse_2
            vb[i] = vb[i] + 2.0 * -__cse_2
            vb[i + 1] = vb[i + 1] + __cse_2
        end
        %MID%
        for i = n - 1:-1:2
            __oldb_0 = vb[i]
            %ZERO%
            ub[i] = ub[i] + __oldb_0
            __cse_3 = 0.25__oldb_0
            ub[i - 1] = ub[i - 1] + __cse_3
            ub[i] = ub[i] + 2.0 * -__cse_3
            ub[i + 1] = ub[i + 1] + __cse_3
        end
    end
    return nothing
end
"""

car_src(; loop = "nsteps:-1:1", mid = "", zero = "vb[i] = 0.0", pre = "") = replace(CAR_BASE, "%LOOP%" => loop, "%MID%" => mid, "%ZERO%" => zero, "%PRE%" => pre)

# (label, source, the fold must fire)
const CAR_UNIT = [
    ("descending time loop", car_src(), true),
    ("ascending time loop", car_src(loop = "1:nsteps"), true),
    ("a time loop with step 2", car_src(loop = "1:2:nsteps"), true),
    ("a time loop with step -2", car_src(loop = "nsteps:-2:1"), true),
    ("the end of the time loop is read from an array (ingest hoists it to a scalar)", car_src(loop = "nsteps:-1:m[1]"), true),
    ("the zeroing scales by a parameter instead of zeroing", car_src(zero = "vb[i] = c * vb[i]"), true),
    ("the zeroing reads a scalar the loop assigns", car_src(mid = "c = 0.5 * t", zero = "vb[i] = c * vb[i]"), false),
    ("the zeroing reads the loop variable", car_src(zero = "vb[i] = 0.0 * t"), false),
    ("the bound of the time loop is assigned in the body", car_src(mid = "nsteps = nsteps - 0"), false),
    ("a scalar statement after the zeroing loop", replace(car_src(), "        end\n    end\n    return nothing" => "        end\n        c = 1.0\n    end\n    return nothing"), false),
]

car_count(s, pat) = length(collect(eachmatch(pat, s)))

function car_unit(src::String; ns = (3, 4, 5, 9, 13), steps = (0, 1, 2, 3, 6))
    expr = Meta.parse(src)
    gk = STADE.cgen_ingest(expr)
    g3 = STADE.cgen_gather_stencil(gk, 3)
    g4 = STADE.cgen_gather_stencil(gk, 4)
    # Julia 1.11 can crash when a collection runs between a fresh eval and its first call: keep the modules alive and the collector off here.
    m0, m4 = Module(), Module()
    GC.enable(false)
    f0 = Core.eval(m0, STADE.cgen_gather_function_expr(gk))
    f4 = Core.eval(m4, STADE.cgen_gather_function_expr(g4))
    names = [string(a) for a in gk.args]
    rng = Random.Xoshiro(STADE.val_stable_seed(src))
    worst = 0.0
    for n in ns, nst in steps
        len = n + 4
        arrs = Dict("ub" => STADE.val_randn(rng, len), "vb" => STADE.val_randn(rng, len))
        scal = Dict{String,Any}("n" => n, "nsteps" => nst, "c" => 0.7, "m" => [1])
        r0, r4 = gat_call(f0, names, arrs, scal), gat_call(f4, names, arrs, scal)
        for k in sort(collect(keys(r0)))
            worst = max(worst, maximum(abs.(r0[k] .- r4[k])))
        end
    end
    GC.enable(true)
    GC.gc()
    t3 = STADE.io_expr_to_source(STADE.cgen_gather_function_expr(g3))
    t4 = STADE.io_expr_to_source(STADE.cgen_gather_function_expr(g4))
    return (worst = worst, fired = t3 != t4, text3 = t3, text4 = t4)
end

function gat_call(f, names, arrs, scal)
    cp = Dict(k => copy(v) for (k, v) in arrs)
    vals = Any[haskey(cp, nm) ? cp[nm] : scal[nm] for nm in names]
    Base.invokelatest(f, vals...)
    return cp
end

function car_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, src, want) in CAR_UNIT
        try
            r = car_unit(src)
            check(r.worst <= 1e-12 && r.fired == want, "$(label): fold $(r.fired ? "fires" : "does not fire") (want $(want ? "fires" : "does not")), difference to the original $(r.worst)")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    # text of the descending case
    r = car_unit(car_src())
    t = r.text4
    check(car_count(t, r"__zf\d+ = min\(abs\(t - nsteps\), 1\)") == 1, "the flag is 0 in the first iteration (distance of the loop variable to its start)")
    check(occursin(r"if __zf\d+ >= 1", t), "the folded statements run only when the flag is 1")
    check(occursin("if nsteps >= 1", t), "the closing zeroing loop runs only if the time loop ran")
    ra = car_unit(car_src(loop = "1:nsteps")).text4
    check(occursin(r"__zf\d+ = min\(abs\(t - 1\), 1\)", ra) && occursin("if 1 <= nsteps", ra), "an ascending loop: flag against its start 1, closing loop if 1 <= nsteps")
    # the other levels keep their text
    check(car_unit(car_src()).text3 == STADE.io_expr_to_source(STADE.cgen_gather_function_expr(STADE.cgen_gather_stencil(STADE.cgen_ingest(Meta.parse(car_src())), 3))), "level 3 is the v0.4.65 text")
    # a kernel without a time loop is the same at level 3 and 4
    plain = "function f(ub, vb, n, c)\n    for i = n - 1:-1:2\n        vb[i] = vb[i] + c * ub[i]\n    end\n    return nothing\nend\n"
    pg = STADE.cgen_ingest(Meta.parse(plain))
    check(STADE.cgen_gather_stencil(pg, 3) == STADE.cgen_gather_stencil(pg, 4), "a kernel without a sequential loop is the same at levels 3 and 4")
    return bad
end

function validate_carry(; sabotage::Bool = true, phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- carry fold checks")
        bad += car_checks()
    end
    if sabotage && :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        repo = abspath(joinpath(@__DIR__, ".."))
        text = read(joinpath(repo, "src", "STADE.jl"), String)
        cases = [
            ("the flag is always 1", "rhs = Expr(:call, :min, Expr(:call, :abs, Expr(:call, :-, S.var, S.lo)), 1))", "rhs = 1)"),
            ("the flag measures against the end of the loop", "Expr(:call, :abs, Expr(:call, :-, S.var, S.lo))", "Expr(:call, :abs, Expr(:call, :-, S.var, S.hi))"),
            ("the folded statements run without the flag", "guard = Expr(:call, :>=, zf, 1)\n", "guard = nothing\n"),
            ("the closing loop runs without the trip test", "ran = (kind = :if, cond = Expr(:call, S.step > 0 ? :<= : :>=, S.lo, S.hi), then = NamedTuple[R], els = NamedTuple[])", "ran = (kind = :if, cond = Expr(:call, :<=, 0, 1), then = NamedTuple[R], els = NamedTuple[])"),
            ("the closing loop tests the wrong direction", "S.step > 0 ? :<= : :>=", "S.step > 0 ? :>= : :<="),
            ("the closing loop is dropped", "return NamedTuple[loop, ran]", "return NamedTuple[loop]"),
            ("a scalar the loop assigns may reach the zeroing loop", "isdisjoint(setdiff(syms, own), assigned) || return nothing\n", ""),
            ("the loop variable may reach the zeroing loop", "    push!(assigned, S.var)\n", ""),
            ("an assigned loop bound is accepted", "    isdisjoint(union(cgen_expr_symbols(S.lo), cgen_expr_symbols(S.hi)), assigned) || return nothing\n", ""),
            ("the zeroing loop need not be last", "    G, R = S.body[1], S.body[end]\n", "    G, R = S.body[1], S.body[2]\n"),
            ("the fold is tried at level 3", "        lv >= 4 && (g = merge(g, (body = cgen_gather_carry_body", "        lv >= 3 && (g = merge(g, (body = cgen_gather_carry_body"),
            ("the zeroing loop stays in the body", "loop = merge(S, (body = vcat(NamedTuple[flag, fused], S.body[2:end - 1]),))", "loop = merge(S, (body = vcat(NamedTuple[flag, fused], S.body[2:end]),))"),
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
                    include($(repr(joinpath(@__DIR__, "validate_carry.jl"))))
                    n = car_checks()
                    println("RESULT ", n)
                """
                out = sab_read(script)
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all carry fold checks passed" : "*** $bad carry fold check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_carry() == 0 ? 0 : 1)
end
