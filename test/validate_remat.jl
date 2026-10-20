include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_remat(dir="val-corpus")

Regression gate for the rematerialized restores of `keep_push_pop = false` (`agen_remat_plan`) and for the branch-scalar hoist guard
that `if_scalar_rewrite` exposed. No GPU needed.

A pre-write snapshot of a scalar is replaced by recomputing the earlier same-block load that produced its old value. That is only
right when the load can be repeated at the backward position with the same result, so each loop of `remat_shapes` breaks one condition:

  loop 1  pure load through an int defined once            -> remat
  loop 2  a self-referential write between two loads       -> the write is remat'd, the load after it is not
  loop 3  load from an array no later write touches        -> remat (y is written only before)
  loop 4  a nested loop rewrites the scalar                -> kept
  loop 5  the earlier definition sits in an `if`           -> kept
  loop 6  a float local stands in the load's operands      -> kept
  loop 7  a read-only float scalar argument in the operands -> remat
  loop 8  the loaded array is rewritten between the loads  -> kept
  loop 9  the loaded array is written earlier in the same enclosing loop -> kept

Three groups of checks:

  1. PLAN: the sites chosen for `remat_shapes` and `so_tile` are exactly the expected right-hand sides.
  2. EFFECT: on `so_tile` the adjoint loses stack references, the restore text appears in the backward sweep, and
     `keep_push_pop = true` still uses push!/pop! stacks (the pass never runs there).
  3. SABOTAGE: in a fresh process each guard is replaced by a version that finds nothing wrong; its kernel (`remat_shapes`, or `if_scalar_rewrite` for the branch hoist
     guard) must then fail its adjoint or dot-product oracle. The enclosing-loop rule for array writes is reported but not required to fail: every array a
     load can come from was also snapshotted at its overwrite, so no shape tried distinguishes it.
"""
function validate_remat(dir::String = joinpath(@__DIR__, "val-corpus"))
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))

    function plan_of(name)
        src = Meta.parse(read(joinpath(dir, name * ".jl"), String))
        kernel = STADE.norm_insert_dead_entry_resets(STADE.parse_kernel(src))
        am = STADE.act_analyze(kernel)
        (_, sites) = STADE.stade_site_level_tbr_check(kernel)
        return sort([string(v) for v in values(STADE.agen_remat_plan(kernel, am, sites))])
    end

    # 1. PLAN
    expected = Dict(
        "remat_shapes" => ["w[base + 1]", "w[ci]", "w[ci] * s", "y[t]"],
        "so_tile" => sort(["w[(ci - 1) * 2 + 1]", "w[(ci - 1) * 2 + 2]", "x[(t + ci) - 1]", "y[(t + ci) - 1]"]),
        "if_scalar_rewrite" => String[],
    )
    for (name, want) in sort(collect(expected))
        got = plan_of(name)
        check(got == want, "plan $name: $(length(got)) site(s) $(got == want ? "as expected" : "got $got, want $want")")
    end

    # 2. EFFECT
    src = Meta.parse(read(joinpath(dir, "so_tile.jl"), String))
    stack_refs(e) = count("_stack[", string(e))
    on = STADE.stade_adjoint(src; keep_push_pop = false, fuse_ii_loops = true)
    kernel = STADE.norm_insert_dead_entry_resets(STADE.parse_kernel(src))
    am = STADE.act_analyze(kernel)
    (sn, ag) = STADE.stade_site_level_tbr_check(kernel)
    off = STADE.agen_emit(kernel, STADE.lin_build(kernel, am), STADE.snap_plan(kernel, am; site_needed = sn);
                          keep_push_pop = false, push_pop = ag, ii_plan = STADE.stade_ii_plan_check(kernel))
    check(stack_refs(on.adjoint) < stack_refs(off.adjoint), "so_tile adjoint: $(stack_refs(off.adjoint)) -> $(stack_refs(on.adjoint)) stack references")
    check(occursin(r"wv = w\[\(ci - 1\) \* 2 \+ 1\]", string(on.adjoint)), "so_tile adjoint: the backward sweep reloads wv instead of popping it")
    kt = STADE.stade_adjoint(src; keep_push_pop = true)
    check(occursin("wv_stack", string(kt.adjoint)) && count("pop!", string(kt.adjoint)) > 0, "keep_push_pop = true: stacks untouched")

    # 3. SABOTAGE
    function sabotaged(def, kernel)
        script = """
            include($(repr(joinpath(@__DIR__, "..", "src", "STADE.jl"))))
            Core.eval(STADE, Meta.parse($(repr(def))))
            path = $(repr(joinpath(dir, kernel * ".jl")))
            a = STADE.stade_validate_adjoint_file(path; trials = 4, keep_push_pop = false, fuse_ii_loops = true)
            d = STADE.stade_validate_dotprod_file(path; trials = 4, keep_push_pop = false, fuse_ii_loops = true)
            println("RESULT ", a.ok, " ", d.ok)
        """
        out = try read(`$(Base.julia_cmd()) --startup-file=no -e $script`, String) catch e; "ERR" end
        m = match(r"RESULT (true|false) (true|false)", out)
        m === nothing && return :error
        return (m.captures[1] == "false" || m.captures[2] == "false") ? :caught : :missed
    end
    purity_head = "function agen_remat_pure(e, kinds, assigned_syms, writes, avail, d_seq, d_chain)\n"
    rec = "all(a -> agen_remat_pure(a, kinds, assigned_syms, writes, avail, d_seq, d_chain), e.args[2:end])"
    sab = [
        ("array loads always pure", purity_head * "e isa Number && return true\ne isa Symbol && return e in avail || !(e in assigned_syms)\ne isa Expr || return false\n(e.head == :call || e.head == :ref) && return $rec\nreturn false\nend", "remat_shapes"),
        ("any symbol pure", purity_head * "e isa Number && return true\ne isa Symbol && return true\ne isa Expr || return false\n(e.head == :call || e.head == :ref) && return $rec\nreturn false\nend", "remat_shapes"),
        ("nested writes never clear the earlier definition", "agen_stmt_assigned!(stmt, acc) = nothing", "remat_shapes"),
        ("nested int writes never void block-local ints", "agen_stmt_assigned(stmt) = Set{Symbol}()", "remat_shapes"),
        ("branch hoist ignores a later write", "agen_stmt_assigned!(stmt, acc) = nothing", "if_scalar_rewrite"),
    ]
    for (nm, def, kernel) in sab
        r = sabotaged(def, kernel)
        check(r == :caught, "sabotage \"$nm\": $kernel $(r == :caught ? "fails its oracle" : String(r))")
    end
    println("\n", bad == 0 ? "all remat checks passed" : "*** $bad remat check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_remat() == 0 ? 0 : 1)
end
