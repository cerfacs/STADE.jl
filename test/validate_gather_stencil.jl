isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# The gather rewrite of a scatter loop (cgen_gather_*).
#
# An adjoint loop that adds into `xb[i + c]` for several literal offsets c can be rewritten so that a thread owns one element and adds the terms
# of iterations j - c, which needs no atomics. The rewrite is a source-to-source step on the ingested statements, so it can be run on the host:
# the rewritten loops are turned back into Julia and compared with the originals, and the rewritten adjoint and HVP of every corpus kernel go
# through the finite-difference and dot-product oracles. A rewrite that fires on no corpus kernel would pass everything, so the number of
# rewritten loops is part of the result.

# the gather loops of a body: the loops the rewrite made, which a residual that is dropped leaves uncounted by a plain loop count
gat_count_gather(body) = sum((s.kind == :for && startswith(string(s.var), "__gj_")) + (s.kind in (:for, :while) ? gat_count_gather(s.body) : s.kind == :if ? gat_count_gather(s.then) + gat_count_gather(s.els) : 0) for s in body; init = 0)
gat_count_for(body) = sum((s.kind == :for) + (s.kind in (:for, :while) ? gat_count_for(s.body) : s.kind == :if ? gat_count_for(s.then) + gat_count_for(s.els) : 0) for s in body; init = 0)

# unit cases: (label, source, expected number of top-level loops after the rewrite). The loop is the only top-level `for` of each source.
const GAT_UNIT = [
    ("scatter with zeroing, 4 offsets, scalar and array coefficients", 2, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        o = ub[i]
        ub[i] = 0.0
        vb[i] = vb[i] + o
        vb[i - 1] = vb[i - 1] + 0.25 * o
        vb[i + 1] = vb[i + 1] + a * o
        vb[i - 2] = vb[i - 2] + o * wb[i]
    end
    return nothing
end"""),
    ("accumulations only, the residual is dropped", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        vb[i - 1] = vb[i - 1] + ub[i]
        vb[i + 2] = vb[i + 2] + 2.0 * ub[i] * wb[i]
    end
    return nothing
end"""),
    ("two receiving arrays with different offset sets", 2, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        o = ub[i]
        ub[i] = 0.0
        vb[i - 1] = vb[i - 1] + o
        vb[i + 1] = vb[i + 1] + 3.0 * o
        wb[i] = wb[i] + a * o
        wb[i + 2] = wb[i + 2] + o * o
    end
    return nothing
end"""),
    ("descending loop", 2, """
function f(ub, vb, wb, a, n)
    for i = n - 2:-1:3
        o = ub[i]
        ub[i] = 0.0
        vb[i - 1] = vb[i - 1] + o
        vb[i + 2] = vb[i + 2] + 2.0 * o
    end
    return nothing
end"""),
    ("scalar chain replayed at several offsets", 2, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        o = ub[i]
        p = o * a
        q = p + wb[i]
        vb[i - 1] = vb[i - 1] + q
        r = q * q
        vb[i + 1] = vb[i + 1] + r * p
    end
    return nothing
end"""),
    ("generated temporaries only: the residual is dropped", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        __t0 = ub[i]
        __t1 = __t0 * a
        vb[i - 1] = vb[i - 1] + __t1
        vb[i + 1] = vb[i + 1] + __t0
    end
    return nothing
end"""),
    ("same offset accumulated twice around a definition", 2, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        o = ub[i]
        ub[i] = 0.0
        vb[i] = vb[i] + o
        q = 2.0 * o
        vb[i] = vb[i] + q
        vb[i + 1] = vb[i + 1] + q
    end
    return nothing
end"""),
    ("receiving array is also read: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        o = vb[i]
        vb[i - 1] = vb[i - 1] + o
        vb[i + 1] = vb[i + 1] + o
    end
    return nothing
end"""),
    ("index with coefficient -1: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        vb[n - i] = vb[n - i] + ub[i]
        vb[i + 1] = vb[i + 1] + ub[i]
    end
    return nothing
end"""),
    ("index with coefficient 2: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        vb[2 * i] = vb[2 * i] + ub[i]
        vb[2 * i + 1] = vb[2 * i + 1] + ub[i]
    end
    return nothing
end"""),
    ("receiving array also written by a plain assignment: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        vb[i + 1] = ub[i]
        vb[i - 1] = vb[i - 1] + ub[i]
    end
    return nothing
end"""),
    ("a scalar read before its definition (carried): refused", 1, """
function f(ub, vb, wb, a, n)
    s = 0.0
    for i = 3:n - 2
        s = s + ub[i]
        vb[i - 1] = vb[i - 1] + s
        vb[i + 1] = vb[i + 1] + 2.0 * s
    end
    return nothing
end"""),
    ("owned array read after its write: scalarized in the replay", 2, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        ub[i] = 0.0
        o = ub[i] + wb[i]
        vb[i - 1] = vb[i - 1] + o
        vb[i + 1] = vb[i + 1] + o
    end
    return nothing
end"""),
    ("a term that reads an owned array after its write", 2, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        ub[i] = 2.0 * ub[i] + a
        vb[i - 1] = vb[i - 1] + ub[i]
        vb[i + 1] = vb[i + 1] + ub[i]
    end
    return nothing
end"""),
    ("owned array accumulated, read, then zeroed", 2, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        ub[i] = ub[i] + wb[i]
        o = ub[i]
        ub[i] = 0.0
        vb[i - 1] = vb[i - 1] + o
        vb[i + 1] = vb[i + 1] + a * o
    end
    return nothing
end"""),
    ("owned array written at i + 1: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        ub[i + 1] = 2.0 * wb[i]
        vb[i - 1] = vb[i - 1] + ub[i]
        vb[i + 1] = vb[i + 1] + ub[i]
    end
    return nothing
end"""),
    ("owned array read at i + 1: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        o = ub[i + 1]
        ub[i] = 0.0
        vb[i - 1] = vb[i - 1] + o
        vb[i + 1] = vb[i + 1] + o
    end
    return nothing
end"""),
    ("step 2: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:2:n - 2
        vb[i - 1] = vb[i - 1] + ub[i]
        vb[i + 1] = vb[i + 1] + ub[i]
    end
    return nothing
end"""),
    ("a branch in the body: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        if a > 0.0
            vb[i - 1] = vb[i - 1] + ub[i]
        end
        vb[i + 1] = vb[i + 1] + ub[i]
    end
    return nothing
end"""),
    ("a nested loop in the body: refused", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        for j = 1:2
            wb[j] = wb[j] + 1.0
        end
        vb[i - 1] = vb[i - 1] + ub[i]
        vb[i + 1] = vb[i + 1] + ub[i]
    end
    return nothing
end"""),
    ("only offset 0: nothing to gather", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        o = ub[i]
        ub[i] = 0.0
        vb[i] = vb[i] + o
    end
    return nothing
end"""),
    ("nine offsets: over the cap", 1, """
function f(ub, vb, wb, a, n)
    for i = 5:n - 4
        vb[i - 4] = vb[i - 4] + ub[i]
        vb[i - 3] = vb[i - 3] + ub[i]
        vb[i - 2] = vb[i - 2] + ub[i]
        vb[i - 1] = vb[i - 1] + ub[i]
        vb[i] = vb[i] + ub[i]
        vb[i + 1] = vb[i + 1] + ub[i]
        vb[i + 2] = vb[i + 2] + ub[i]
        vb[i + 3] = vb[i + 3] + ub[i]
        vb[i + 4] = vb[i + 4] + ub[i]
        wb[i] = 2.0 * ub[i]
    end
    return nothing
end"""),
    ("a replay of 26 statements: over the cap", 1, """
function f(ub, vb, wb, a, n)
    for i = 3:n - 2
        s1 = ub[i] * 1.01
        s2 = s1 * 1.01
        s3 = s2 * 1.01
        s4 = s3 * 1.01
        s5 = s4 * 1.01
        s6 = s5 * 1.01
        s7 = s6 * 1.01
        s8 = s7 * 1.01
        s9 = s8 * 1.01
        s10 = s9 * 1.01
        s11 = s10 * 1.01
        s12 = s11 * 1.01
        s13 = s12 * 1.01
        s14 = s13 * 1.01
        s15 = s14 * 1.01
        s16 = s15 * 1.01
        s17 = s16 * 1.01
        s18 = s17 * 1.01
        s19 = s18 * 1.01
        s20 = s19 * 1.01
        s21 = s20 * 1.01
        s22 = s21 * 1.01
        s23 = s22 * 1.01
        s24 = s23 * 1.01
        s25 = s24 * 1.01
        vb[i - 1] = vb[i - 1] + s25
        vb[i + 1] = vb[i + 1] + s25
        wb[i] = 2.0 * ub[i]
    end
    return nothing
end"""),
]

# run the original and the rewritten function on the same random arrays for several n, zero-trip ones included; returns the loops after the
# rewrite and the largest difference
function gat_run_unit(src::String; ns = (0, 1, 2, 3, 4, 5, 6, 9, 13, 20))
    expr = Meta.parse(src)
    gk = STADE.cgen_ingest(expr)
    gk2 = STADE.cgen_gather_stencil(gk, true)
    loops = count(s -> s.kind == :for, gk2.body)
    m1, m2 = Module(), Module()
    f1 = Core.eval(m1, STADE.cgen_gather_function_expr(gk))
    f2 = Core.eval(m2, STADE.cgen_gather_function_expr(gk2))
    rng = MersenneTestRNG(src)
    worst = 0.0
    for n in ns
        len = 2 * n + 12
        base = [randn(rng, len) for _ in 1:3]
        a = randn(rng)
        r1 = [copy(b) for b in base]
        r2 = [copy(b) for b in base]
        Base.invokelatest(f1, r1..., a, n)
        Base.invokelatest(f2, r2..., a, n)
        for k in 1:3
            worst = max(worst, maximum(abs.(r1[k] .- r2[k])))
        end
    end
    # Without this, the unit run died with a segmentation fault (in Julia's type cache) on 3 of 4 runs once 22 cases had each defined a function in two
    # anonymous modules; with it, 12 of 12 runs pass. The cause was not identified, and nothing in the generated code touches memory outside its arrays.
    GC.gc()
    return loops, worst
end
MersenneTestRNG(src) = MersenneTwister(hash(src))

function gat_unit_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, want, src) in GAT_UNIT
        try
            loops, worst = gat_run_unit(src)
            check(loops == want && worst <= 1e-12, "$(label): $(loops) loop(s) (want $(want)), max difference $(worst)")
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    # an empty loop whose gather range is wide must not touch an array shorter than that range
    src = """
function f(ub, vb, n)
    for i = 4:2
        vb[i - 1] = vb[i - 1] + ub[i]
        vb[i + 3] = vb[i + 3] + ub[i]
    end
    return nothing
end"""
    ok = try
        gk = STADE.cgen_ingest(Meta.parse(src))
        gk2 = STADE.cgen_gather_stencil(gk, true)
        f2 = Core.eval(Module(), STADE.cgen_gather_function_expr(gk2))
        Base.invokelatest(f2, zeros(3), zeros(3), 3)
        count(s -> s.kind == :for, gk2.body) == 1
    catch
        false
    end
    check(ok, "empty loop with a wide gather range and a 3-element array: no access")
    return bad
end


# The zero-fold: a loop that touches only its own elements (the residual of a gather, or a plain zeroing loop), folded into the gather loop that
# follows it. (label, loops without the fold, loops with it, source). A loop that is refused must come back unchanged.
const GAT_FUSE_UNIT = [
    ("the ping-pong adjoint: the zeroing of ub joins the second gather", 4, 3, """
function f(ub, vb, a, n)
    for i = n - 1:-1:2
        o = ub[i]
        ub[i] = 0.0
        vb[i] = vb[i] + o
        vb[i - 1] = vb[i - 1] + 0.25 * o
        vb[i + 1] = vb[i + 1] + 0.25 * o
    end
    for i = n - 1:-1:2
        o = vb[i]
        vb[i] = 0.0
        ub[i] = ub[i] + o
        ub[i - 1] = ub[i - 1] + 0.25 * o
        ub[i + 1] = ub[i + 1] + 0.25 * o
    end
    return nothing
end"""),
    ("a zeroing loop before a gather into another array", 3, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        o = ub[i]
        ub[i] = 0.0
        vb[i] = vb[i] + o
        vb[i - 1] = vb[i - 1] + 0.25 * o
    end
    for i = 2:n - 1
        wb[i - 1] = wb[i - 1] + a * vb[i]
        wb[i + 1] = wb[i + 1] + vb[i]
    end
    return nothing
end"""),
    ("a plain zeroing loop before a gather into the same array", 2, 1, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        ub[i] = 0.0
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i] = ub[i] + 2.0 * wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a zeroing loop that two later gathers could take: only the next one", 3, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        ub[i] = 0.0
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + 2.0 * wb[i]
        ub[i + 1] = ub[i + 1] + 2.0 * wb[i]
    end
    return nothing
end"""),
    ("the gather reads the zeroed array at another index: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        ub[i] = 0.0
    end
    for i = 2:n - 1
        wb[i - 1] = wb[i - 1] + ub[i]
        wb[i + 1] = wb[i + 1] + ub[i]
    end
    return nothing
end"""),
    ("the gather reads the zeroed array through a scalar chain: refused", 4, 4, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        o = ub[i]
        ub[i] = 0.0
        vb[i] = vb[i] + o
        vb[i - 1] = vb[i - 1] + o
    end
    for i = 2:n - 1
        p = ub[i] * a
        wb[i - 1] = wb[i - 1] + p
        wb[i + 1] = wb[i + 1] + 2.0 * p
    end
    return nothing
end"""),
    ("the zeroed range is wider than the gather range: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 1:n
        ub[i] = 0.0
    end
    for i = 3:n - 2
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("the zeroed range is inside the gather range", 2, 1, """
function f(ub, vb, wb, a, n)
    for i = 4:n - 3
        ub[i] = 0.0
    end
    for i = 3:n - 2
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a scalar of the zeroing loop is read after the gather: refused", 3, 3, """
function f(ub, vb, wb, a, n)
    o = 0.0
    for i = 2:n - 1
        o = ub[i]
        ub[i] = 0.0
        vb[i] = vb[i] + o
        vb[i - 1] = vb[i - 1] + o
    end
    for i = 2:n - 1
        wb[i - 1] = wb[i - 1] + vb[i]
        wb[i + 1] = wb[i + 1] + vb[i]
    end
    wb[1] = o
    return nothing
end"""),
    ("a zeroing loop that carries a scalar: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    s = 0.0
    for i = 2:n - 1
        s = s + ub[i]
        ub[i] = s
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a zeroing loop that reads a neighbour: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        ub[i] = 0.5 * ub[i - 1]
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a zeroing loop that writes at an offset: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        ub[i + 1] = 0.0
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a statement between the two loops: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        ub[i] = 0.0
    end
    wb[1] = a
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a zeroing loop with a step of 2: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:2:n - 1
        ub[i] = 0.0
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a zeroing loop with a step of 2 and constant bounds, inside the gather range: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 4:2:10
        ub[i] = 0.0
    end
    for i = 2:11
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a zeroing loop with a branch: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        if ub[i] > 0.0
            ub[i] = 0.0
        end
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("a zeroing loop that also writes an array the gather accumulates into at another index: refused", 2, 2, """
function f(ub, vb, wb, a, n)
    for i = 2:n - 1
        wb[i] = 0.0
        ub[i] = 0.0
    end
    for i = 2:n - 1
        ub[i - 1] = ub[i - 1] + wb[i]
        ub[i + 1] = ub[i + 1] + wb[i]
    end
    return nothing
end"""),
    ("the zeroing loop sits inside a time loop (the time loop counts as one)", 5, 4, """
function f(ub, vb, a, n, nt)
    for t = 1:nt
        for i = n - 1:-1:2
            o = ub[i]
            ub[i] = 0.0
            vb[i] = vb[i] + o
            vb[i - 1] = vb[i - 1] + 0.25 * o
            vb[i + 1] = vb[i + 1] + 0.25 * o
        end
        for i = n - 1:-1:2
            o = vb[i]
            vb[i] = 0.0
            ub[i] = ub[i] + o
            ub[i - 1] = ub[i - 1] + 0.25 * o
            ub[i + 1] = ub[i + 1] + 0.25 * o
        end
    end
    return nothing
end"""),
]

# the loops of a body at any depth (a fold removes one per fused pair)
gat_loops_deep(body) = gat_count_for(body)

# run the same function at gather level 1 (no fold) and level 2 (fold) and the original: the loop counts, the largest difference to the original, and
# whether the two rewritten functions agree to the last bit

# one call of a rewritten function on copies of the arrays; the copies come back
function gat_call(f, names, arrs, scal)
    cp = Dict(k => copy(v) for (k, v) in arrs)
    vals = Any[haskey(cp, nm) ? cp[nm] : scal[nm] for nm in names]
    Base.invokelatest(f, vals...)
    return cp
end

function gat_run_fuse_unit(src::String; ns = (0, 1, 2, 3, 4, 5, 6, 9, 13, 20), nargs = 3)
    expr = Meta.parse(src)
    gk = STADE.cgen_ingest(expr)
    g1 = STADE.cgen_gather_stencil(gk, 1)
    g2 = STADE.cgen_gather_stencil(gk, 2)
    m0, m1, m2 = Module(), Module(), Module()
    f0 = Core.eval(m0, STADE.cgen_gather_function_expr(gk))
    f1 = Core.eval(m1, STADE.cgen_gather_function_expr(g1))
    f2 = Core.eval(m2, STADE.cgen_gather_function_expr(g2))
    names = [string(a) for a in gk.args]
    rng = MersenneTwister(hash(src))
    worst = 0.0
    exact = true
    for n in ns
        len = 2 * n + 12
        arrs = Dict("ub" => randn(rng, len), "vb" => randn(rng, len), "wb" => randn(rng, len))
        scal = Dict{String,Any}("a" => randn(rng), "n" => n, "nt" => 3)
        r0, r1, r2 = gat_call(f0, names, arrs, scal), gat_call(f1, names, arrs, scal), gat_call(f2, names, arrs, scal)
        for k in sort(collect(keys(r0)))
            worst = max(worst, maximum(abs.(r0[k] .- r2[k])))
            exact &= all(isequal.(r1[k], r2[k]))
        end
    end
    GC.gc()
    return gat_loops_deep(g1.body), gat_loops_deep(g2.body), worst, exact
end

function gat_fuse_checks()
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    for (label, want1, want2, src) in GAT_FUSE_UNIT
        try
            l1, l2, worst, exact = gat_run_fuse_unit(src)
            if startswith(label, "the ping-pong adjoint")
                folded_src = STADE.io_expr_to_source(STADE.cgen_gather_function_expr(STADE.cgen_gather_stencil(STADE.cgen_ingest(Meta.parse(src)), 2)))
                check(!occursin("_z", folded_src) && occursin("ub[__gj_2] = 0.0", folded_src), "$(label): the folded statements are the zeroing alone (the temporaries nobody reads are dropped)")
            end
            check(l1 == want1 && l2 == want2 && worst <= 1e-12 && exact, "$(label): $(l1) loops without the fold (want $(want1)), $(l2) with it (want $(want2)), difference to the original $(worst), folded and unfolded " * (exact ? "identical" : "DIFFER"))
        catch e
            check(false, "$(label): error $(first(split(sprint(showerror, e), "\n")))")
        end
    end
    return bad
end

# every kernel of the corpus, adjoint and HVP, with and without fusion: the rewritten code must pass the oracles. Returns (failures, rewritten loops by kernel)
function gat_kernel_check(path::String, fuse::Bool, mode::Symbol, scratch::String; int_lo::Int = 3, int_hi::Int = 5)
    name = splitext(basename(path))[1]
    primal_expr = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal_expr)
    yp = joinpath(scratch, name * ".yaml")
    Random.seed!(hash(name))
    isfile(yp) || STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = int_lo, int_hi = int_hi)
    baseline = STADE.io_read_baseline_yaml(yp)
    STADE.val_coerce_int_arrays!(kernel, baseline.values)
    adj = STADE.stade_adjoint(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse)
    stack_arg_names = Symbol[a for a in adj.initstacks.args[1].args[2:end]]
    nrew = Ref(0)
    nfold = Ref(0)
    function rewrite(e)
        gk = STADE.cgen_ingest(e)
        gk2 = STADE.cgen_gather_stencil(gk, true)
        nrew[] += gat_count_gather(gk2.body) - gat_count_gather(gk.body)
        nfold[] += gat_loops_deep(STADE.cgen_gather_stencil(gk, 1).body) - gat_loops_deep(gk2.body)
        return STADE.cgen_gather_function_expr(gk2)
    end
    adj2 = merge(adj, (adjoint = rewrite(adj.adjoint),))
    if mode == :adjoint
        r = STADE.val_validate_adjoint(kernel, primal_expr, adj2, baseline; trials = 3, epsilon = 1e-6, rtol = 1e-3, stack_arg_names = stack_arg_names)
    else
        hv = STADE.stade_hvp(primal_expr; keep_push_pop = false, fuse_ii_loops = fuse)
        hv2 = merge(hv, (hvp = rewrite(hv.hvp),))
        r = STADE.val_validate_hvp(kernel, primal_expr, adj2, hv2, baseline; trials = 3, epsilon = 1e-6, rtol = 1e-3, stack_arg_names = stack_arg_names)
    end
    return r.ok, nrew[], nfold[]
end

# kernels that must be rewritten (loops added, in the adjoint) and kernels that must not be
const GAT_MUST = ["gat_sum_scalar", "gat_pingpong", "gat_diff1", "gat_wide", "gat_two_targets", "gat_accum_out", "gat_reverse", "gat_nonlinear"]
# kernels whose adjoint has a zeroing loop that folds into the next gather
const GAT_FOLD_MUST = ["gat_pingpong", "lin_pingpong", "lin_quotient", "lin_then_quad"]
const GAT_MUST_NOT = ["gat_inplace_seq", "gat_stride", "gat_reduce_scalar", "gat_alias", "gat_rows", "gat_const_idx"]

function gat_corpus_checks(dir::String; names = nothing, fuses = (false, true), modes = (:adjoint, :hvp), int_lo::Int = 3, int_hi::Int = 5, report = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    files = sort(filter(f -> endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl")), readdir(dir)))
    names === nothing || (files = [f for f in files if splitext(f)[1] in names])
    rewritten = Dict{String,Int}()
    folded = Dict{String,Int}()
    mktempdir() do scratch
        for f in files
            name = splitext(f)[1]
            for fuse in fuses, mode in modes
                ok, n, nf = try
                    gat_kernel_check(joinpath(dir, f), fuse, mode, scratch; int_lo, int_hi)
                catch e
                    println("  !! ", name, " ", first(split(sprint(showerror, e), "\n"))[1:min(end, 160)])
                    (false, 0, 0)
                end
                rewritten[name] = max(get(rewritten, name, 0), n)
                folded[name] = max(get(folded, name, 0), nf)
                if report
                    check(ok, "$(rpad(name, 28)) [$(mode), fuse=$(Int(fuse))] oracles pass, $(n) loop(s) added by the rewrite")
                elseif !ok
                    check(false, "$(rpad(name, 28)) [$(mode), fuse=$(Int(fuse))] an oracle fails with the rewrite applied")
                end
            end
        end
    end
    for k in GAT_MUST
        (names === nothing || k in names) || continue
        check(get(rewritten, k, 0) > 0, "$(k): the rewrite fires ($(get(rewritten, k, 0)) loops added)")
    end
    for k in GAT_MUST_NOT
        (names === nothing || k in names) || continue
        check(get(rewritten, k, 0) == 0, "$(k): the rewrite refuses ($(get(rewritten, k, 0)) loops added)")
    end
    for k in GAT_FOLD_MUST
        (names === nothing || k in names) || continue
        check(get(folded, k, 0) > 0, "$(k): the zero-fold fires ($(get(folded, k, 0)) loops folded)")
    end
    if names === nothing
        total = count(>(0), values(rewritten))
        check(total >= 10, "$(total) of $(length(rewritten)) corpus kernels have a rewritten loop (need at least 10, or the rewrite is not exercised)")
    end
    return bad
end

"""
    validate_gather_stencil(dir = "val-corpus"; sabotage = true)

  1. UNIT: 23 loops through the rewrite (offsets of both signs, several receiving arrays, a descending loop, replayed scalar chains, owned arrays kept in scalars, and 12
     shapes that must be refused). The original and the rewritten function run on the same random arrays for n from 0 to 20, so zero-trip loops
     are included. A loop that is refused must come back as one loop.
  1b. ZERO-FOLD: 18 sources through gather level 1 and level 2 (6 in which the zeroing of a received array joins the next gather loop, and 12 shapes that
     must not fold). The loop counts are fixed (level 2 has one loop fewer per fold) and the level 2 function must
     agree with the level 1 function to the last bit (`isequal`) and with the original to rounding.
  2. CORPUS: every kernel, adjoint and HVP, with and without fusion, with the gather rewrite applied to the generated code, must pass the
     finite-difference oracle. 8 `gat_*` kernels must be rewritten and 6 must be refused. At least 10 kernels of the corpus must have a rewritten
     loop. `gat_pingpong`, `lin_pingpong`, `lin_quotient` and `lin_then_quad` must fold.
  3. SABOTAGE: 37 edits of the rewrite and of the fold (offset sign, each guard, each end of the gather range, orientation, the write flag,
     each refusal condition, the residual, each condition and guard of the fold) must fail a check. Every anchor must occur exactly once in
     `STADE.jl`. `phases` and `slice` split the run (a full run takes about 30 minutes).
"""
function validate_gather_stencil(dir::String = joinpath(@__DIR__, "val-corpus"); sabotage::Bool = true, phases = (:units, :corpus, :sabotage), slice = 1:typemax(Int))
    bad = 0
    if :units in phases
        println("-- unit checks")
        bad += gat_unit_checks()
        println("-- zero-fold checks")
        bad += gat_fuse_checks()
    end
    if :corpus in phases
        println("-- corpus")
        bad += gat_corpus_checks(dir; report = false)
    end
    if sabotage && :sabotage in phases
        println("-- sabotage")
        check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
        repo = abspath(joinpath(@__DIR__, ".."))
        text = read(joinpath(repo, "src", "STADE.jl"), String)
        cases = [
            ("offset sign of the shifted index", "rhs = add(jv, -c))]", "rhs = add(jv, c))]"),
            ("lower guard ignores the offset", "cond = Expr(:call, :>=, jv, add(amin, c))", "cond = Expr(:call, :>=, jv, add(amin, 0))"),
            ("upper guard ignores the offset", "cond = Expr(:call, :<=, jv, add(amax, c))", "cond = Expr(:call, :<=, jv, add(amax, 0))"),
            ("gather range misses the low offsets", "lo = add(amin, offsets[1])", "lo = amin"),
            ("gather range misses the high offsets", "hi = add(amax, offsets[end])", "hi = amax"),
            ("orientation of a descending loop ignored", "amin, amax = stmt.step == 1 ? (first_val, last_val) : (last_val, first_val)", "amin, amax = (first_val, last_val)"),
            ("write flag always true", "cond = Expr(:call, :>, Symbol(\"__gh_\", arr), 0)", "cond = Expr(:call, :>=, Symbol(\"__gh_\", arr), 0)"),
            ("receiving array may be read", "if isempty(r) && all(x -> x[2] && cgen_gather_offset(x[3], var) !== nothing, w)", "if all(x -> x[2] && cgen_gather_offset(x[3], var) !== nothing, w)"),
            ("receiving array may be plainly written", "if isempty(r) && all(x -> x[2] && cgen_gather_offset(x[3], var) !== nothing, w)", "if isempty(r) && all(x -> cgen_gather_offset(x[3], var) !== nothing, w)"),
            ("owned array may be written off the loop index", "all(x -> x[3] === var, w) || return nothing", "true"),
            ("owned array may be read off the loop index", "all(x -> x[2] === var, r) || return nothing", "true"),
            ("owned element not loaded at the top of a replay", "own_name(arr) in needed && push!(blk,", "false && push!(blk,"),
            ("owned plain write not replayed", "elseif !haskey(accs, k) && s.lhs.args[1] in oset", "elseif false && s.lhs.args[1] in oset"),
            ("owned accumulation replayed as a plain write", "prog[k] = (nm, Expr(:call, :+, nm, own_ren(accs[k][3])))", "prog[k] = (nm, own_ren(accs[k][3]))"),
            ("owned reads not renamed in a gathered term", "terms = Dict{Int,Any}(k => (a[1], cgen_gather_offset(a[2], var), own_ren(a[3]))", "terms = Dict{Int,Any}(k => (a[1], cgen_gather_offset(a[2], var), a[3])"),
            ("scalars may be read before they are defined", "for r in reads\n            (r in scalars && !(r in defined)) && return nothing", "for r in reads\n            false && return nothing"),
            ("any step is accepted", "(stmt.step isa Integer && abs(stmt.step) == 1) || return nothing", "stmt.step isa Integer || return nothing"),
            ("offset cap removed", "length(offsets) <= 8 || return nothing", "true"),
            ("replay cap removed", "length(keep) <= 24 || return nothing", "true"),
            ("residual always dropped", "residual_dead = !any(s -> s.lhs isa Expr, rest) &&", "residual_dead = true || !any(s -> s.lhs isa Expr, rest) &&"),
            ("residual always kept", "residual_dead && return NamedTuple[gather]", "false && return NamedTuple[gather]"),
            ("scalar definitions not replayed", "union!(needed, intersect(cgen_read_vars(terms[k][3]), scalars))", "nothing"),
            ("branches in the body accepted", "all(s -> s.kind == :assign && !cgen_is_stackop_assign(s) && !cgen_is_stack_alloc(s.rhs) && !cgen_is_length_call(s.rhs), body) || return nothing", "all(s -> !cgen_is_stackop_assign(s) && !cgen_is_stack_alloc(s.rhs) && !cgen_is_length_call(s.rhs), body) || return nothing"),
            ("fold: the zeroed range may be wider than the gather range", "    cgen_gather_ge(amin, G.lo) && cgen_gather_ge(G.hi, amax) || return nothing\n", ""),
            ("fold: a scalar that is read later may be folded", "    isempty(locals) || isempty(ii_escapes_nested(root, R.body, locals)) || return nothing\n", ""),
            ("fold: the gather may read a zeroed array at another index", "        (arr in wr && !(length(idxs) == 1 && idxs[1] === G.var)) && return nothing\n", ""),
            ("fold: a carried scalar is accepted", "        for r in cgen_read_vars(s.rhs)\n            (r in scalars && !(r in defined)) && return nothing\n        end\n", ""),
            ("fold: a read at a neighbouring index is accepted", "            (length(idxs) == 1 && idxs[1] === var) || return nothing\n            push!(rd, arr)", "            push!(rd, arr)"),
            ("fold: a write at an offset is accepted", "        (l isa Expr && l.head == :ref && length(l.args) == 2 && l.args[1] isa Symbol && l.args[2] === var) || return nothing\n            push!(wr, l.args[1])", "        push!(wr, l.args[1])"),
            ("fold: any step is accepted", "    (st.step isa Integer && abs(st.step) == 1) || return nothing\n    isempty(st.body) && return nothing", "    isempty(st.body) && return nothing"),
            ("fold: a branch in the zeroed loop is accepted", "    all(s -> s.kind == :assign && !cgen_is_stackop_assign(s) && !cgen_is_stack_alloc(s.rhs) && !cgen_is_length_call(s.rhs), st.body) || return nothing", "    all(s -> !cgen_is_stackop_assign(s) && !cgen_is_stack_alloc(s.rhs) && !cgen_is_length_call(s.rhs), st.body) || return nothing"),
            ("fold: the upper guard is dropped", "inner = (kind = :if, cond = Expr(:call, :<=, G.var, amax), then = stmts, els = NamedTuple[])", "inner = (kind = :if, cond = Expr(:call, :<=, 0, 1), then = stmts, els = NamedTuple[])"),
            ("fold: the lower guard is dropped", "outer = (kind = :if, cond = Expr(:call, :>=, G.var, amin), then = NamedTuple[inner], els = NamedTuple[])", "outer = (kind = :if, cond = Expr(:call, :>=, 1, 0), then = NamedTuple[inner], els = NamedTuple[])"),
            ("fold: the zeroing runs after the gather's own write", "return merge(G, (body = vcat(NamedTuple[outer], G.body),))", "return merge(G, (body = vcat(G.body, NamedTuple[outer]),))"),
            ("fold: temporaries nobody reads are kept", "        if s.lhs isa Symbol && !(s.lhs in live)\n            continue\n        end\n", ""),
            ("fold: level 1 folds too", "    lv >= 2 || return g\n", ""),
            ("fold: level 2 does not fold", "    lv >= 2 || return g\n", "    return g\n"),
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
                    include($(repr(joinpath(@__DIR__, "validate_gather_stencil.jl"))))
                    n = gat_unit_checks()
                    n += gat_fuse_checks()
                    n += gat_corpus_checks($(repr(dir)); names = ["gat_pingpong", "gat_diff1", "gat_two_targets", "gat_reverse", "gat_wide", "lin_pingpong"], modes = (:adjoint,), report = false)
                    println("RESULT ", n)
                """
                out = try read(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch e; "ERR" end
                m = match(r"RESULT (\d+)", out)
                caught = out == "ERR" || m === nothing || parse(Int, m.captures[1]) > 0
                check(caught, "sabotage \"$label\": " * (caught ? "a check fails" : "NOT CAUGHT"))
            end
        end
    end
    println("\n", bad == 0 ? "all gather stencil checks passed" : "*** $bad gather stencil check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_gather_stencil() == 0 ? 0 : 1)
end
