include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_if_nesting(dir = "val-corpus")

Regression gate for four defects found by mutating the corpus in 0.4.10 (three wrapped a loop body one level deeper, the fourth moved a stage ahead of a level loop). No GPU needed.

  * Branch flags under keep_push_pop = true. The forward sweep pushed an `if`'s flag at the start of its arm and the backward sweep pops it
    before the arm, so every `if` inside an `if` read the wrong flag. The flag is now pushed after the arm's body.
  * Branch scalars. The backward sweep recomputed a scalar assigned in an `if` from its right-hand side after restoring it from its stack,
    reading arrays in a later state. That recompute (the branch-scalar hoist) was removed: the head-of-body restore already covers it.
  * Retiring assignments in the sizing pass of keep_push_pop = false. The split point of a ragged block looked at top-level statements only,
    so a retiring assignment nested in an `if` or a loop left every block's table one value late.
  * Size terms of early stages under keep_push_pop = false. The slots of a ragged block start after the slots of the stages before it, and
    a stage's size read a scalar (`nc`) that a later loop reassigns. The forward sweep read the live value and the backward sweep a stale
    one, so the two sweeps used different indexes. The sizing pass of initstacks_* now captures such a term (`__mult_<k>`) where it stands.

Three groups of checks:

  1. WITNESSES: each corpus witness passes the adjoint and dot-product oracles in all four (keep_push_pop, fuse_ii_loops) combinations.
  2. REFUSAL: a ragged block that reads its bound again after reassigning it stops with a message for keep_push_pop = false, and still
     generates a correct adjoint for keep_push_pop = true.
  3. SABOTAGE: in a fresh process each fix is reverted by patching one place of the source; its witness must then fail an oracle (or, for
     the refusal, the generator must stop refusing).
"""
function validate_if_nesting(dir::String = joinpath(@__DIR__, "val-corpus"))
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))

    # 1. WITNESSES
    witnesses = ["nest_if", "nest_if_scalar", "if_scalar_then_if", "if_scalar_then_loop_if", "inplace_stencil_if", "retire_in_if",
                 "retire_if", "retire_nested", "if_top_scalar", "if_top_two", "if_in_outer_loop", "early_stage_offset"]
    for name in witnesses, (kpp, fuse) in ((true, false), (true, true), (false, false), (false, true))
        path = joinpath(dir, name * ".jl")
        a = STADE.stade_validate_adjoint_file(path; trials = 4, keep_push_pop = kpp, fuse_ii_loops = fuse)
        d = STADE.stade_validate_dotprod_file(path; trials = 4, keep_push_pop = kpp, fuse_ii_loops = fuse)
        check(a.ok && d.ok, "$name keep_push_pop=$kpp fuse_ii_loops=$fuse")
    end

    # 2. REFUSAL
    reuse = joinpath(mktempdir(), "retire_reuse.jl")
    write(reuse, """
        function retire_reuse(x, y, n, levels, out)
            cur = n
            for i_l = 1:levels
                for i = 1:cur
                    if n >= 0
                        t = x[i] * x[i]
                        y[i] = y[i] + t * t
                    end
                end
                cur = div(cur + 1, 2)
                for i = 1:cur
                    if n >= 0
                        t = x[i] * x[i]
                        y[i] = y[i] + t * t
                    end
                end
            end
            out[1] = y[1]
            return nothing
        end
        """)
    msg = try
        STADE.stade_adjoint(Meta.parse(read(reuse, String)); keep_push_pop = false)
        ""
    catch e
        sprint(showerror, e)
    end
    check(occursin("reads the bound `cur` in a loop header after it is reassigned", msg), "keep_push_pop = false refuses a bound read after its reassignment")
    d = STADE.stade_validate_dotprod_file(reuse; trials = 4, keep_push_pop = true, fuse_ii_loops = false)
    check(d.ok, "keep_push_pop = true still differentiates it correctly")

    # 3. SABOTAGE
    stade_src = read(joinpath(@__DIR__, "..", "src", "STADE.jl"), String)
    function sabotaged(patches, path, kpp; gen_only = false)
        text = stade_src
        for (old, new) in patches
            count(old, text) == 1 || return :error
            text = replace(text, old => new)
        end
        tmp = tempname() * ".jl"
        write(tmp, text)
        script = """
            include($(repr(tmp)))
            path = $(repr(path))
            gen_ok = try STADE.stade_adjoint(Meta.parse(read(path, String)); keep_push_pop = $kpp, fuse_ii_loops = false); true catch; false end
            if $gen_only
                println("GEN ", gen_ok)
                exit()
            end
            gen_ok || (println("GENFAIL"); exit())
            function both(path)
                ok1 = ok2 = true
                for fuse in (true, false)
                    try
                        a = STADE.stade_validate_adjoint_file(path; trials = 4, keep_push_pop = $kpp, fuse_ii_loops = fuse)
                        d = STADE.stade_validate_dotprod_file(path; trials = 4, keep_push_pop = $kpp, fuse_ii_loops = fuse)
                        ok1 &= a.ok; ok2 &= d.ok
                    catch
                        ok1 = ok2 = false
                    end
                end
                return (ok1, ok2)
            end
            res = both(path)
            println("RESULT ", res[1], " ", res[2])
        """
        out = try read(`$(Base.julia_cmd()) --startup-file=no -e $script`, String) catch e; "ERR" end
        rm(tmp; force = true)
        gen_only && return occursin("GEN true", out) ? :generated : :refused
        occursin("GENFAIL", out) && return :caught
        m = match(r"RESULT (true|false) (true|false)", out)
        m === nothing && return :error
        return (m.captures[1] == "false" || m.captures[2] == "false") ? :caught : :missed
    end

    # the flag goes back to the start of the then-arm, in front of the arm's body
    arm_body = join(["                append!(then_exprs, agen_forward_body(stmt.then, kinds, active_map, value_needed, reassigned, stacks, exempt;",
                     "                                                        ectx = ectx,",
                     "                                                        lin_body = lin_body === nothing ? nothing : lin_body[idx].then,",
                     "                                                        unsafe = unsafe))"], "\n")
    push_flag = "                agen_emit_push!(then_exprs, nm, 1, ectx, key)"
    flag_after = arm_body * "\n" * push_flag
    flag_before = push_flag * "\n" * arm_body
    # a retiring assignment counts only when it is itself a top-level statement
    split_line = "        if retire\n            assigned = Set{Symbol}()"
    split_top = "        if retire && st.kind == :assign\n            assigned = Set{Symbol}()"
    # a bound read after its reassignment is no longer refused
    refuse_line = "elseif !isempty(intersect(e[2], assigned))"
    refuse_off = "elseif false"
    # an early stage's size term is read live again
    cap_line = "        if !isempty(intersect(vs, later))"
    cap_off = "        if false"

    for (nm, patches, kernel, kpp) in [
            ("branch flag pushed before its arm", [(flag_after, flag_before)], "nest_if", true),
            ("branch flag pushed before its arm", [(flag_after, flag_before)], "nest_if_scalar", true),
            ("retiring assignment only at top level", [(split_line, split_top)], "retire_in_if", false),
            ("early stage size not captured", [(cap_line, cap_off)], "early_stage_offset", false)]
        r = sabotaged(patches, joinpath(dir, kernel * ".jl"), kpp)
        check(r == :caught, "sabotage \"$nm\": $kernel fails its oracle" * (r == :caught ? "" : " -- got $r"))
    end
    r = sabotaged([(refuse_line, refuse_off)], reuse, false; gen_only = true)
    check(r == :generated, "sabotage \"no refusal\": the generator no longer stops" * (r == :generated ? "" : " -- got $r"))

    println(bad == 0 ? "\nall if-nesting checks passed" : "\n$(bad) NOT OK")
    return bad
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_if_nesting() == 0 ? 0 : 1)
end
