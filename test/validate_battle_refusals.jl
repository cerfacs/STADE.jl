# Battle test, refusals: a shape STADE cannot offload or differentiate must be refused with a message that names the cause.
# A refusal that turns into a silent acceptance, or whose message names nothing, is a finding.
include(joinpath(@__DIR__, "validate_battle_status.jl"))

function validate_battle_refusals()
    bad = 0; findings = 0
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    chk(name, cond, detail = "") = (cond ? println("ok      ", name) : (bad += 1; println("NOT OK  ", name, " ", detail)); cond)
    note(name, detail) = (findings += 1; println("FINDING ", name, ": ", detail))
    # 1 while in adjoint and hvp with keep_push_pop = false (plan decision: refuse with a clear message)
    for f in bt_subject_files(["f6_"])
        prim = STADE.io_read_corpus_entry(joinpath(BT_DIR, f))
        for (mode, g) in ((:adjoint, p -> STADE.stade_adjoint(p; keep_push_pop = false, fuse_ii_loops = true)),
                          (:hvp, p -> STADE.stade_hvp(p; keep_push_pop = false, fuse_ii_loops = true)))
            m = msg(() -> g(prim))
            chk("$f $mode refused", !isempty(m))
            chk("$f $mode message names the cause", occursin("while", m) || occursin("restored", m) || occursin("trip count", m), first(m, 120))
        end
        # the supported mode must still work
        m = msg(() -> STADE.stade_adjoint(prim; keep_push_pop = true, fuse_ii_loops = false))
        chk("$f adjoint with keep_push_pop = true works", isempty(m), first(m, 120))
    end
    # 2 elseif: the direct path and the file path
    src = "function ei(x, y, n)\n for i1 = 1:n\n if x[i1] > 1.0\n y[i1] = 1.0\n elseif x[i1] > 0.0\n y[i1] = x[i1]\n else\n y[i1] = 0.0\n end\n end\n return nothing\n end\n"
    fe = bt_parse_fn(src)
    m1 = msg(() -> STADE.stade_tangent(fe))
    chk("elseif refused on the direct path", !isempty(m1))
    chk("elseif message names elseif (direct path)", occursin("elseif", m1), first(m1, 120))
    tmp = mktempdir(); p = joinpath(tmp, "ei.jl"); write(p, src)
    m2 = msg(() -> STADE.io_read_corpus_entry(p))
    chk("elseif refused on the file path", !isempty(m2))
    occursin("elseif", m2) || note("elseif message on the file path", "refused with: " * first(m2, 100) * " (the direct path says: " * first(m1, 100) * ")")
    # 3 a plan for a body the stages cannot lower must not be a silent host copy of everything with kernels = 0 and no loop kept:
    #   every subject of F1 must produce at least one kernel on both backends in primal mode
    for f in bt_subject_files(["f1_"])
        prim = STADE.io_read_corpus_entry(joinpath(BT_DIR, f))
        f in ("f1_sumsq.jl",) && continue            # reduction into a scalar: see the status family
        for be in (:cuda, :jacc)
            pl = bt_plan(prim, be, :pinned)
            chk("$f primal $be has a kernel", length(pl.kernels) >= 1)
        end
    end
    println(bad == 0 ? "battle refusals: all checks pass, $findings finding(s)" : "battle refusals: $bad NOT OK, $findings finding(s)")
    return bad
end
if abspath(PROGRAM_FILE) == @__FILE__
    validate_battle_refusals() == 0 || exit(1)
end
