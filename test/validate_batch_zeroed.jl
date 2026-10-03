include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# bgen_array_left_zeroed is an optimization: it proves the fill! in
# bgen_reset_local! unnecessary for a buffer the sweep leaves at zero.
#
# Its branches cannot be reached from a corpus kernel. A :local buffer in
# a generated adjoint is accumulated and then zeroed, and every loop bound
# is a runtime value. Rule 11 settles the second: if the bound is zero
# neither loop runs, so the buffer keeps whatever the caller left, and no
# coverage analysis can prove otherwise. The bodies below are therefore
# built by hand, so every branch is executed rather than assumed.

asg(lhs, rhs) = (kind = :assign, lhs = lhs, rhs = rhs)
fr(v, lo, hi, step, body) = (kind = :for, var = v, lo = lo, hi = hi, step = step, body = body)
iff(c, t, e) = (kind = :if, cond = c, then = t, els = e)
ref(a, i) = Expr(:ref, a, i)

function validate_batch_zeroed(dir::String = joinpath(@__DIR__, "val-corpus"))
    checks = 0; bad = 0
    function expect(label, want::Bool, body)
        checks += 1
        got = STADE.bgen_array_left_zeroed(NamedTuple[body...], :vb)
        if got == want
            println(rpad(label, 52), " ok  ", got)
        else
            bad += 1
            println(rpad(label, 52), " FAIL  got ", got, " want ", want)
        end
    end

    # the one shape it can prove: literal bounds, every write a zeroing write
    expect("literal-bound zeroing loop", true,
        [fr(:i, 1, 3, 1, NamedTuple[asg(ref(:vb, :i), 0.0)])])

    # rule 11: a loop that may not run has not run
    expect("literal but empty loop (1:0)", false,
        [fr(:i, 1, 0, 1, NamedTuple[asg(ref(:vb, :i), 0.0)])])
    expect("runtime bound proves nothing", false,
        [fr(:i, 1, :n, 1, NamedTuple[asg(ref(:vb, :i), 0.0)])])

    # coverage: a single element zeroed is not the array zeroed
    expect("one element zeroed at top level", false,
        [asg(ref(:vb, 3), 0.0)])

    # a non-zero write anywhere defeats the proof, even when a zeroing
    # loop follows it -- this is the real generated-adjoint shape
    expect("accumulate then zero, both literal", false,
        [fr(:i, 1, 3, 1, NamedTuple[asg(ref(:vb, :i), Expr(:call, :+, ref(:vb, :i), 1.0))]),
         fr(:j, 3, 1, -1, NamedTuple[asg(ref(:vb, :j), 0.0)])])

    # rule 11 in reverse: an unprovable write destroys an earlier fact
    expect("literal zero then runtime-bound zero", false,
        [fr(:i, 1, 3, 1, NamedTuple[asg(ref(:vb, :i), 0.0)]),
         fr(:j, 1, :n, 1, NamedTuple[asg(ref(:vb, :j), 0.0)])])

    # both arms of an if must zero it
    expect("if zeroes in both arms", true,
        [iff(:c, NamedTuple[fr(:i, 1, 3, 1, NamedTuple[asg(ref(:vb, :i), 0.0)])],
                NamedTuple[fr(:i, 1, 2, 1, NamedTuple[asg(ref(:vb, :i), 0.0)])])])
    expect("if zeroes in one arm only", false,
        [iff(:c, NamedTuple[fr(:i, 1, 3, 1, NamedTuple[asg(ref(:vb, :i), 0.0)])],
                NamedTuple[])])

    # a whole-array assignment is not an indexed zeroing write
    expect("scalar assignment to the array name", false,
        [asg(:vb, 0.0)])

    # an index that only mentions a loop variable leaves an element behind
    expect("index is loopvar + 1", false,
        [fr(:i, 1, 3, 1, NamedTuple[asg(ref(:vb, Expr(:call, :+, :i, 1)), 0.0)])])

    # ---- the measurement that decides whether to wire it in ----
    subjects = [("ddp_affine", [:u], Symbol[]), ("ddp_twoparam", [:u], Symbol[]),
                ("ddp_ragged", [:u], Symbol[]), ("ddp_branch", [:u], Symbol[]),
                ("ddp_tangent_input", [:u], Symbol[]), ("ddp_batch_hist", [:u], [:hist]),
                ("affine_loss", [:u], Symbol[]), ("matvec_loss", [:u, :v], Symbol[])]
    tot = 0; prov = 0
    for (name, ps, rd) in subjects
        p = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        k = STADE.parse_kernel(p)
        ep = STADE.stade_batch(p; per_sample = ps, mode = :adjoint, reduced = rd)
        out = STADE.stade_adjoint(p; keep_push_pop = false, fuse_ii_loops = true)
        ak = STADE.cgen_ingest(out.adjoint)
        for b in STADE.bgen_role_args(k, ep.roles, :adjoint, :local)
            tot += 1
            STADE.bgen_array_left_zeroed(ak.body, b) && (prov += 1)
        end
    end
    checks += 1
    if prov == 0
        println(rpad("no corpus :local buffer is provably zeroed", 52), " ok  0/", tot)
    else
        bad += 1
        println(rpad("no corpus :local buffer is provably zeroed", 52), " FAIL  ", prov, "/", tot,
                " -- wiring this into bgen_reset_local! is now worth doing")
    end

    println("\n", checks - bad, "/", checks, " checks passed",
            bad == 0 ? "" : "   *** $bad NOT OK ***")
    return bad
end

validate_batch_zeroed()
