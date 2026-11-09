include(joinpath(@__DIR__, "sabotage_run.jl"))
# ==================== validate_gather_periodic.jl =====================
# Checks the periodic gather (gather_stencil level 3) and the range reduction of the periodic neighbour (cgen_mod_reduce).
#   - units: scatter loops whose target is a periodic shift of the cell (1, 2 and 3 coordinates, shifts of one and two steps, several terms on one permutation, a reset scalar)
#     run before and after the rewrite on random arrays for n from 0 to 9 (n = 1 and n = 2 make the shifts coincide); the rewrite must agree to rounding
#   - refusals: loops that look like a periodic scatter and are not (a transposed index, another modulus, a clamp, an unused coordinate, a carried scalar, a loop that
#     does not start at 1 or does not cover the box); each must come back with no gather loop and the same result
#   - range reduction: the forms for shifts 0, -1 and -2; the other shifts, a coordinate that is reassigned, and another modulus must keep `mod`
#   - SABOTAGE: edits of the new code, each of which one check must catch.
#   julia validate_gather_periodic.jl        units, sabotage
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

gp_count_gather(body) = sum((s.kind == :for && startswith(string(s.var), "__gj_")) + (s.kind in (:for, :while) ? gp_count_gather(s.body) : s.kind == :if ? gp_count_gather(s.then) + gp_count_gather(s.els) : 0) for s in body; init = 0)
gp_text(gk) = STADE.io_expr_to_source(STADE.cgen_gather_function_expr(gk))
gp_count_mod(gk) = length(collect(eachmatch(r"mod\(", gp_text(gk))))

const GP_HEAT3D = """function heat3d_step(u, un, n)
    nn = n * n
    for idx = 1:n * nn
        i = mod(idx - 1, n) + 1
        j = mod(div(idx - 1, n), n) + 1
        k = div(idx - 1, nn) + 1
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        jm = mod(j - 2, n) + 1
        jp = mod(j, n) + 1
        km = mod(k - 2, n) + 1
        kp = mod(k, n) + 1
        c = u[idx]
        un[idx] = c + 0.1 * (u[(k - 1) * nn + (j - 1) * n + im] + u[(k - 1) * nn + (j - 1) * n + ip] + u[(k - 1) * nn + (jm - 1) * n + i] + u[(k - 1) * nn + (jp - 1) * n + i] + u[(km - 1) * nn + (j - 1) * n + i] + u[(kp - 1) * nn + (j - 1) * n + i] - 6.0 * c)
    end
    return nothing
end"""

# (label, source, expected number of gather loops at level 3, size of the arrays as a function of n)
# every source has the arguments (ub, vb, n) and loops over a box of n x n cells
gp_loop(extra, head, body) = """function f(ub, vb, n, m)
$(extra)    for $(head)
        i = mod(idx - 1, n) + 1
        j = div(idx - 1, n) + 1
$(body)
    end
    return nothing
end"""

const GP_UNIT = [
    ("two coordinates, a shift in each, the loop descends", 1, gp_loop("", "idx = n * n:-1:1", """
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        jm = mod(j - 2, n) + 1
        o = vb[idx]
        vb[idx] = 0.0
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + 0.3 * o
        ub[(jm - 1) * n + ip] = ub[(jm - 1) * n + ip] + 0.2 * o
        ub[idx] = ub[idx] + o""")),
    ("shifts of two steps and of +1, read-only coefficient array", 1, gp_loop("", "idx = 1:n * n", """
        i2 = mod(i - 3, n) + 1
        j2 = mod(j, n) + 1
        o = vb[idx]
        ub[(j - 1) * n + i2] = ub[(j - 1) * n + i2] + o
        ub[(j2 - 1) * n + i] = ub[(j2 - 1) * n + i] + 2.0 * o""")),
    ("two terms on the same permutation", 1, gp_loop("", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        o = vb[idx]
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + o
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + 3.0 * o * o""")),
    ("a scalar that is reset at the end of the iteration", 1, gp_loop("    cb = 0.0\n", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        o = vb[idx]
        cb = cb + o
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + 0.5 * o
        cb = cb * 2.0
        o2 = cb
        cb = 0.0
        ub[idx] = ub[idx] + o2""")),
    # refusals: the loop comes back with no gather loop
    ("a transposed index is refused", 0, gp_loop("", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        o = vb[idx]
        ub[(i - 1) * n + (mod(j - 2, n) + 1)] = ub[(i - 1) * n + (mod(j - 2, n) + 1)] + o
        ub[idx] = ub[idx] + o""")),
    ("another modulus is refused", 0, gp_loop("", "idx = 1:n * n", """
        jm = mod(j - 2, m) + 1
        o = vb[idx]
        ub[(jm - 1) * n + i] = ub[(jm - 1) * n + i] + o""")),
    ("a clamp is not a periodic shift", 0, gp_loop("", "idx = 1:n * n", """
        jm = min(max(j - 1, 1), n)
        o = vb[idx]
        ub[(jm - 1) * n + i] = ub[(jm - 1) * n + i] + o""")),
    ("a target that leaves out a coordinate is refused", 0, gp_loop("", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        o = vb[idx]
        ub[im] = ub[im] + o""")),
    ("a carried scalar (never reset) is refused", 0, gp_loop("    s = 0.0\n", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        s = s + vb[idx]
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + s""")),
    ("a loop that starts at 2 is refused", 0, gp_loop("", "idx = 2:n * n", """
        im = mod(i - 2, n) + 1
        o = vb[idx]
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + o""")),
    ("a loop that stops before the end of the box is refused", 0, gp_loop("", "idx = 1:n * n - 1", """
        im = mod(i - 2, n) + 1
        o = vb[idx]
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + o""")),
    ("a target that is not shifted at all keeps the plain loop", 0, gp_loop("", "idx = 1:n * n", """
        o = vb[idx]
        ub[(j - 1) * n + i] = ub[(j - 1) * n + i] + o""")),
    ("a stride that is not the coordinate's is refused", 0, gp_loop("", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        o = vb[idx]
        ub[(j - 1) * m + im] = ub[(j - 1) * m + im] + o""")),
    ("a neighbour of a middle coordinate with another modulus is refused", 0, gp_loop("", "idx = 1:n * n", """
        im = mod(i - 2, m) + 1
        o = vb[idx]
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + o""")),
    ("a reset that an `if` after it spoils in another loop is refused", 0, """
function f(ub, vb, n, m)
    cb = 0.0
    for idx = 1:n * n
        o = vb[idx]
        cb = cb + o
        cb = 0.0
        if o > 0.0
            cb = 1.0
        end
    end
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        j = div(idx - 1, n) + 1
        im = mod(i - 2, n) + 1
        o = vb[idx]
        cb = cb + o
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + 0.5 * cb
        cb = 0.0
    end
    return nothing
end"""),
    ("the index scalars assigned twice with other temporaries, an alias and a scalar-held index (an adjoint that recomputes the forward values)", 1, """
function f(ub, vb, n, m)
    for idx = 1:n * n
        c0 = idx - 1
        c1 = mod(c0, n) + 1
        i = c1
        c2 = div(c0, n) + 1
        j = c2
        im = mod(i - 2, n) + 1
        c3 = (j - 1) * n
        iw = c3 + im
        o = vb[idx]
        t = o * 2.0 + vb[iw]
        i = c1
        j = c2
        im = mod(i - 2, n) + 1
        c4 = (j - 1) * n
        iw = c4 + im
        ub[iw] = ub[iw] + 0.5 * o
        ub[idx] = ub[idx] + t
    end
    return nothing
end"""),
    ("an index held in a scalar, assigned once", 1, gp_loop("", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        iw = (j - 1) * n + im
        o = vb[idx]
        ub[iw] = ub[iw] + 0.5 * o""")),
    ("a neighbour scalar assigned twice with different expressions is refused", 0, gp_loop("", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        o = vb[idx]
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + o
        im = mod(i, n) + 1
        ub[(j - 1) * n + im] = ub[(j - 1) * n + im] + 2.0 * o""")),
    ("a scalar index that depends on a scalar assigned twice differently is refused", 0, gp_loop("", "idx = 1:n * n", """
        im = mod(i - 2, n) + 1
        iw = (j - 1) * n + im
        o = vb[idx]
        ub[iw] = ub[iw] + o
        im = mod(i, n) + 1
        ub[iw] = ub[iw] + 2.0 * o""")),
]

function gp_set_reduce(flag::Bool)  # the debugging switch; the default is on
    Core.eval(STADE, :(cgen_mod_reduce_enabled() = $flag))
end

# the original and the rewritten function on the same random arrays; returns (gather loops, largest difference)
function gp_run(gk; level::Int = 3, reduce::Bool = true, ns = (0, 1, 2, 3, 4, 5, 7, 9), arrays = 2, three = false)
    gp_set_reduce(reduce)
    try
        g = Base.invokelatest(STADE.cgen_gather_stencil, gk, level)
        m0, m1 = Module(), Module()
        f0 = Core.eval(m0, STADE.cgen_gather_function_expr(gk))
        f1 = Core.eval(m1, STADE.cgen_gather_function_expr(g))
        rng = MersenneTwister(hash(string(gk.name, level, reduce)))
        worst = 0.0
        for n in ns
            len = three ? n^3 : n * n
            base = [randn(rng, len) for _ in 1:max(arrays, 1)]
            a0 = Any[[copy(b) for b in base]..., n, n]
            a1 = Any[[copy(b) for b in base]..., n, n]
            three && (a0 = a0[1:end-1]; a1 = a1[1:end-1])
            Base.invokelatest(f0, a0...); Base.invokelatest(f1, a1...)
            for k in 1:arrays
                worst = max(worst, isempty(base[k]) ? 0.0 : maximum(abs.(a0[k] .- a1[k])))
            end
        end
        GC.gc()
        return gp_count_gather(g.body), worst
    finally
        gp_set_reduce(false)
    end
end

# the heat3d adjoint (arrays ub, vb, ... of n^3 cells and n) before and after the rewrite
function gp_adj_run(gk, lv::Int, reduce::Bool)
    gp_set_reduce(reduce)
    local cnt::Int = 0
    local w::Float64 = 0.0
    try
        g = Base.invokelatest(STADE.cgen_gather_stencil, gk, lv)
        # the modules are kept alive: a function of a collected module crashes the runtime
        m0, m1 = Module(), Module()
        f0 = Core.eval(m0, STADE.cgen_gather_function_expr(gk))
        f1 = Core.eval(m1, STADE.cgen_gather_function_expr(g))
        narr = length(gk.args) - 1
        GC.@preserve m0 m1 for n in (1, 2, 3, 4, 5)
            rng = MersenneTwister(n)
            base = [randn(rng, n^3 + 64 * n^2 + 64) for _ in 1:narr]  # stack arrays of an adjoint need up to a few dozen slots per cell
            a0 = Any[[copy(b) for b in base]..., n]
            a1 = Any[[copy(b) for b in base]..., n]
            Base.invokelatest(f0, a0...); Base.invokelatest(f1, a1...)
            for k in 1:narr
                w = max(w, maximum(abs.(a0[k] .- a1[k])))
            end
        end
        cnt = gp_count_gather(g.body)
    finally
        gp_set_reduce(false)
    end
    return cnt, w
end

const GP_BURGERS = """function burgers2d_step(u, v, un, vn, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        j = div(idx - 1, n) + 1
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        jm = mod(j - 2, n) + 1
        jp = mod(j, n) + 1
        iw = (j - 1) * n + im
        ie = (j - 1) * n + ip
        is = (jm - 1) * n + i
        inn = (jp - 1) * n + i
        uc = u[idx]
        vc = v[idx]
        # upwind weights: the positive part of the speed looks west/south, the negative part east/north
        upos = max(uc, 0.0)
        uneg = min(uc, 0.0)
        vpos = max(vc, 0.0)
        vneg = min(vc, 0.0)
        un[idx] = uc - 0.1 * (upos * (uc - u[iw]) + uneg * (u[ie] - uc) + vpos * (uc - u[is]) + vneg * (u[inn] - uc)) + 0.05 * (u[iw] + u[ie] + u[is] + u[inn] - 4.0 * uc)
        vn[idx] = vc - 0.1 * (upos * (vc - v[iw]) + uneg * (v[ie] - vc) + vpos * (vc - v[is]) + vneg * (v[inn] - vc)) + 0.05 * (v[iw] + v[ie] + v[is] + v[inn] - 4.0 * vc)
    end
    return nothing
end"""

const GP_PROBE6 = """function capprobe_q006(u, un, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        j = div(idx - 1, n) + 1
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        jm = mod(j - 2, n) + 1
        jp = mod(j, n) + 1
        iw = (j - 1) * n + im
        ie = (j - 1) * n + ip
        is = (jm - 1) * n + i
        inn = (jp - 1) * n + i
        uc = u[idx]
        a = u[iw]
        b = u[ie]
        c = u[is]
        d = u[inn]
        s0 = 0.5 * uc + 0.1 * (a + b + c + d)
        # chain of dependent polynomial updates, one scalar per step
        s1 = s0 + 0.1 * a * (1.0 - s0 * s0)
        s2 = s1 + 0.1 * b * (1.0 - s1 * s1)
        s3 = s2 + 0.1 * c * (1.0 - s2 * s2)
        s4 = s3 + 0.1 * d * (1.0 - s3 * s3)
        s5 = s4 + 0.1 * uc * (1.0 - s4 * s4)
        s6 = s5 + 0.1 * a * (1.0 - s5 * s5)
        un[idx] = s6
    end
    return nothing
end"""

function gp_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, want, src) in GP_UNIT
        for reduce in (false, true)
            try
                gk = STADE.cgen_ingest(Meta.parse(src))
                loops, worst = gp_run(gk; reduce)
                check(loops == want && worst <= 1e-12, "$(label) [reduce=$(Int(reduce))]: $(loops) gather loop(s) (want $(want)), max difference $(worst)")
            catch e
                check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
            end
        end
    end
    # the cap on the replayed statements: the first source with the limit at 1 statement is refused, at the default it gathers
    capsrc = GP_UNIT[findfirst(x -> startswith(x[1], "the index scalars assigned twice"), GP_UNIT)][3]
    for (lim, want) in (("1", 0), (nothing, 1))
        old = get(ENV, "STADE_GATHER_CAP", nothing)
        lim === nothing ? delete!(ENV, "STADE_GATHER_CAP") : (ENV["STADE_GATHER_CAP"] = lim)
        try
            loops, worst = gp_run(STADE.cgen_ingest(Meta.parse(capsrc)))
            check(loops == want && worst <= 1e-12, "replay cap $(lim === nothing ? "default" : lim): $(loops) gather loop(s) (want $(want)), max difference $(worst)")
        finally
            old === nothing ? delete!(ENV, "STADE_GATHER_CAP") : (ENV["STADE_GATHER_CAP"] = old)
        end
    end
    # the cost gate: replayed work against the work of the scatter. burgers2d replays little beyond the body and gathers; a chain of 6 dependent
    # statements is replayed by all five offsets, costs about 4 times the scatter, and is refused (the ratio switch lets it through)
    check(STADE.cgen_gather_worth(10, 100, 5) && !STADE.cgen_gather_worth(1000, 100, 5), "cost gate: 10 against 100 passes, 1000 against 100 is refused")
    check(STADE.cgen_gather_worth(120, 100, 5) && !STADE.cgen_gather_worth(121, 100, 5), "cost gate: the threshold is 0.8 of the body plus 10 per atomic")
    for (lbl, src, ratio, want) in (("burgers2d", GP_BURGERS, nothing, 1), ("chain of 6", GP_PROBE6, nothing, 0), ("chain of 6 with the ratio at 100", GP_PROBE6, "100", 1))
        old = get(ENV, "STADE_GATHER_RATIO", nothing)
        ratio === nothing ? delete!(ENV, "STADE_GATHER_RATIO") : (ENV["STADE_GATHER_RATIO"] = ratio)
        try
            adj = STADE.stade_adjoint(Meta.parse(src); keep_push_pop = false, fuse_ii_loops = true)
            gk = STADE.cgen_ingest(Meta.parse(STADE.io_expr_to_source(adj.adjoint)))
            loops, worst = gp_adj_run(gk, 3, true)
            check(loops == want && worst <= 1e-12, "cost gate, $(lbl): $(loops) gather loop(s) (want $(want)), max difference $(worst)")
        finally
            old === nothing ? delete!(ENV, "STADE_GATHER_RATIO") : (ENV["STADE_GATHER_RATIO"] = old)
        end
    end
    # the heat equation adjoint in three coordinates: 6 permutations and the cell itself, fires at level 3 and not at level 2
    adj = STADE.stade_adjoint(Meta.parse(GP_HEAT3D); keep_push_pop = false, fuse_ii_loops = true)
    gk3 = STADE.cgen_ingest(Meta.parse(STADE.io_expr_to_source(adj.adjoint)))
    for reduce in (false, true)
        for (lv, want) in ((2, 0), (3, 1))
            loops, worst = gp_adj_run(gk3, lv, reduce)
            check(loops == want && worst <= 1e-12, "heat3d adjoint, level $(lv) [reduce=$(Int(reduce))]: $(loops) gather loop(s) (want $(want)), max difference $(worst)")
        end
    end
    return bad
end

const GP_MOD = [
    # (label, source, mods left after the reduction)
    ("shifts 0, -1 and -2 are reduced, +1 and -3 are not", 3, """
function f(out, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        a = mod(i, n) + 1
        b = mod(i - 1, n) + 1
        c = mod(i - 2, n) + 1
        d = mod(i - 3, n) + 1
        e = mod(i + 1, n) + 1
        out[idx] = 1000.0 * a + 100.0 * b + 10.0 * c + d + 0.1 * e
    end
    return nothing
end"""),
    ("a coordinate that is reassigned is not reduced", 2, """
function f(out, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        i = i + 1
        c = mod(i - 2, n) + 1
        out[idx] = 10.0 * c + i
    end
    return nothing
end"""),
    ("another modulus is not reduced", 2, """
function f(out, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        c = mod(i - 2, n + 0) + 1
        out[idx] = 10.0 * c + i
    end
    return nothing
end"""),
    ("a coordinate assigned inside a branch is not reduced after it", 2, """
function f(out, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        if idx > 3
            i = 1
        end
        c = mod(i - 2, n) + 1
        out[idx] = 10.0 * c + i
    end
    return nothing
end"""),
    ("a neighbour inside a branch is reduced", 1, """
function f(out, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        c = 1
        if idx > 3
            c = mod(i - 2, n) + 1
        end
        out[idx] = 10.0 * c + i
    end
    return nothing
end"""),
    ("a coordinate reassigned at the end of a loop body is not reduced at its top", 2, """
function f(out, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        for t = 1:3
            c = mod(i - 2, n) + 1
            out[idx] = out[idx] + 10.0 * c
            i = i + 1
        end
    end
    return nothing
end"""),
]

function gp_mod_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, want_mods, src) in GP_MOD
        try
            gk = STADE.cgen_ingest(Meta.parse(src))
            gp_set_reduce(true)
            g = Base.invokelatest(STADE.cgen_gather_stencil, gk, 3)
            gp_set_reduce(false)
            m0, m1 = Module(), Module()
            f0 = Core.eval(m0, STADE.cgen_gather_function_expr(gk)); f1 = Core.eval(m1, STADE.cgen_gather_function_expr(g))
            worst = 0.0
            for n in (1, 2, 3, 4, 5, 8, 9)
                o0 = zeros(n * n); o1 = zeros(n * n)
                Base.invokelatest(f0, o0, n); Base.invokelatest(f1, o1, n)
                worst = max(worst, isempty(o0) ? 0.0 : maximum(abs.(o0 .- o1)))
            end
            mods = gp_count_mod(g)
            check(mods == want_mods && worst == 0.0, "$(label): $(mods) mod call(s) left (want $(want_mods)), max difference $(worst)")
        catch e
            gp_set_reduce(false)
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    return bad
end

function validate_gather_periodic(; phases = (:units, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- units")
        bad += gp_unit_checks()
        println("-- range reduction")
        bad += gp_mod_checks()
    end
    if :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        text = read(joinpath(abspath(joinpath(@__DIR__, "..")), "src", "STADE.jl"), String)
        cases = [
            ("the inverse shifts the wrong way", "inv = call(:+, call(:mod, call(:-, tc, 1 + sh), E), 1)", "inv = call(:+, call(:mod, call(:-, tc, 1 - sh), E), 1)"),
            ("the one-step inverse of a shift of -1 is that of +1", "d = sh == -1 ? call(:-, 1, call(:*, E, call(:>=, tc, E))) : call(:+, -1, call(:*, E, call(:<, tc, 2)))", "d = sh == 1 ? call(:-, 1, call(:*, E, call(:>=, tc, E))) : call(:+, -1, call(:*, E, call(:<, tc, 2)))"),
            ("the loop may start above 1", "    amin == 1 || return nothing\n", ""),
            ("the box is not compared with the loop length", "    sort(total) == fac(amax) || return nothing\n", ""),
            ("the stride of a term is not compared", "(stride == 1 ? coords[c][1] == 1 : (coords[c][1] != 1 && fac(stride) == fac(coords[c][1]))) || return nothing", "true"),
            ("the modulus of a neighbour is not compared", "slot[c][3] === nothing || fac(slot[c][3]) == fac(E) || return nothing", "true"),
            ("an unshifted target is gathered", "    any(c -> slot[c][2] != 0, order) || return nothing\n", ""),
            ("a scalar that is not reset is taken as reset", "ok = ctx !== nothing && haskey(lit, nm) && get(ctx, nm, nothing) == lit[nm]", "ok = true"),
            ("a reset scalar is not set at the top of the replay", "isempty(rbd) || (body = vcat(NamedTuple[(kind = :assign, lhs = r, rhs = ctx.resets[r]) for r in sort(rbd)], body))", ""),
            ("an assignment after the reset is ignored", "st.lhs isa Symbol && (last[st.lhs] = nothing)", "nothing"),
            ("the shift 0 form uses a strict comparison", "a == 0 && return call(:-, call(:+, c, 1), call(:*, E, call(:>=, c, E)))", "a == 0 && return call(:-, call(:+, c, 1), call(:*, E, call(:>, c, E)))"),
            ("the shift -2 form wraps at the wrong place", "a == -2 && return call(:+, call(:-, c, 1), call(:*, E, call(:<, c, 2)))", "a == -2 && return call(:+, call(:-, c, 1), call(:*, E, call(:<, c, 1)))"),
            ("the shift -1 form adds one", "a == -1 && return c\n", "a == -1 && return call(:+, c, 1)\n"),
            ("a reassigned coordinate keeps its bound", "                delete!(env, st.lhs)\n", ""),
            ("a coordinate assigned in a branch keeps its bound", "            inner = Dict{Symbol,Any}(k => v for (k, v) in env if !(k in killed) && isdisjoint(cgen_expr_symbols(v), killed))\n", "            inner = copy(env)\n"),
            ("a modulus that differs from the coordinate's is reduced", "                        Ec == E || continue\n", ""),
            ("a scalar assigned twice differently is taken as stable", "(!any(isnothing, cs) && all(==(cs[1]), cs)) && (r = cs[1])", "(!any(isnothing, cs)) && (r = cs[1])"),
            ("an alias scalar is not resolved", "defs[nm] = cgen_subst_expr(defs[nm], alias)", "nothing"),
            ("a scalar-held index is not inlined", "    idxe = inline_idx(idxe, 0)\n", ""),
            ("the replay cap is ignored", "length(keep) <= cgen_gather_cap() || return nothing", "true"),
            ("the cost gate is removed", "cgen_gather_worth(gather_work, body_work, length(terms)) || return nothing", "true"),
            ("the replayed work of one offset only is counted", "gather_work += sum(", "gather_work = sum("),
            ("the atomics are free in the scatter work", "(body_work + 10.0 * nterms)", "(body_work + 0.0 * nterms)"),
        ]
        mktempdir() do tmp
            for (label, old, new) in cases[intersect(eachindex(cases), slice)]
                n = length(findall(old, text))
                if n != 1
                    check(false, "sabotage \"$label\": anchor found $n times")
                    continue
                end
                get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && (check(true, "anchor \"$label\": found once"); continue)
                path = joinpath(tmp, "STADE_sab.jl")
                write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_gather_periodic.jl"))))
                    n = sab_phase_sum($(repr("validate_gather_periodic.jl")), $(repr(label)), [gp_unit_checks, gp_mod_checks])
                    println("RESULT ", n)
                """
                out = sab_read(script)
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all periodic gather checks passed" : "*** $bad periodic gather check(s) failed ***")
    return bad
end
abspath(PROGRAM_FILE) == abspath(joinpath(@__DIR__, "validate_gather_periodic.jl")) && validate_gather_periodic(; phases = (:units, :sabotage))
