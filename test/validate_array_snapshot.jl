isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# the plan of one kernel as "arrays/loops", or "-" when no loop qualifies; `ii` supplies fuse_ii_loops' classification
function asnap_plan_string(path::String; ii::Bool = false)
    kernel = STADE.norm_insert_dead_entry_resets(STADE.parse_kernel(STADE.io_read_corpus_entry(path)))
    am = STADE.act_analyze(kernel)
    (_, sites) = STADE.stade_site_level_tbr_check(kernel)
    (sites, _) = STADE.agen_remat_apply(kernel, am, sites)
    (_, plan) = STADE.agen_asnap_plan(kernel, am, sites; ii_plan = ii ? STADE.stade_ii_plan_check(kernel) : nothing)
    return isempty(plan.arrays) ? "-" : join(string.(plan.arrays), ",") * "/" * string(length(plan.bodies))
end

# every array argument after a call, primal and adjoint, plus the scalar adjoints the call returned; stack arguments are left out
function asnap_state(kernel, args, ret)
    d = Dict{Symbol,Vector{Float64}}(:__scalar_adjoints => ret === nothing ? Float64[] : ret isa Tuple ? Float64[x for x in ret] : Float64[ret])
    pos = 1
    for a in kernel.sig.args
        k = kernel.sig.kinds[a]
        if k == :scalar_int
            pos += 1
        elseif k in (:scalar_float, :array_float)
            if k == :array_float
                d[a] = vec(copy(args[pos]))
                d[Symbol(string(a), "_shadow")] = vec(copy(args[pos+1]))
            end
            pos += 2
        else
            pos += 1
        end
    end
    return d
end
asnap_maxdiff(a, b) = maximum((maximum(abs.(a[k] .- b[k]); init = 0.0) for k in keys(a)); init = 0.0)
asnap_scale(a) = max(1.0, maximum((maximum(abs.(v); init = 0.0) for v in values(a)); init = 0.0))

# `asnap_ok` with a synthetic ii plan: first marks the Jacobi loop, then its cell loop. Both must leave no snapshot.
function asnap_synthetic_ii(path::String)
    kernel = STADE.norm_insert_dead_entry_resets(STADE.parse_kernel(STADE.io_read_corpus_entry(path)))
    am = STADE.act_analyze(kernel)
    (_, sites) = STADE.stade_site_level_tbr_check(kernel)
    jac = findfirst(s -> s.kind == :for, kernel.body)
    cell = findfirst(s -> s.kind == :for && s.var == :i_cell, kernel.body[jac].body)
    outer = Dict{Any,Symbol}(STADE.agen_site_key(kernel.body, jac) => :recompute)
    inner = Dict{Any,Symbol}(STADE.agen_site_key(kernel.body[jac].body, cell) => :recompute)
    empty(ii) = isempty(STADE.agen_asnap_plan(kernel, am, sites; ii_plan = ii)[2].arrays)
    return (empty(outer), empty(inner))
end

"""
    validate_array_snapshot(dir="val-corpus")

Regression gate for the array snapshot of `keep_push_pop = false` (`agen_asnap_plan`). No GPU needed.

A loop that scatters into an array `A` and reads `A` after every write would normally push the old value at every write, and the push
reads the old value, which keeps the loop on the host. The snapshot copies the whole of `A` once at the head of each iteration and
restores it once at the tail of the reverse of that iteration. It is right only when every derivative read of `A` in the body follows
every write to `A`, so each guard below refuses one shape.

Five groups of checks:

  1. PLAN: the arrays chosen for each kernel are exactly the expected ones, and no other corpus kernel gets a snapshot.
  2. EFFECT: on `ttgc` the stack of per-write saves is gone, the snapshot stack exists, and `keep_push_pop = true` is untouched.
  3. STATE: the final value of every array, primal and adjoint, equals what `keep_push_pop = true` leaves. A gradient can stay
     bit-identical while the primal array is left wrong, so the gradient oracles alone are not enough. A negative control removes the
     restore loop and must change the state.
  4. GUARDS: the guards that no kernel can break at run time (an ii-classified loop, the plan with a synthetic ii plan) are tested
     on the plan itself.
  5. SABOTAGE: in a fresh process each guard is replaced by a version that finds nothing wrong. The kernel written to break it must
     then fail an oracle, or its array state must change.
"""
function validate_array_snapshot(dir::String = joinpath(@__DIR__, "val-corpus"))
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    kernels = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") && !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl"))])

    function kernel_of(name)
        src = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        return (src, STADE.norm_insert_dead_entry_resets(STADE.parse_kernel(src)))
    end
    plan_of(name; ii = false) = asnap_plan_string(joinpath(dir, name * ".jl"); ii = ii)

    # 1. PLAN
    expected = Dict(
        "asnap_ok" => "m/1", "asnap_2d" => "M/1", "asnap_ifwrite" => "m/1", "asnap_twoarrays" => "m1,m2/1",
        "asnap_zerocover" => "m/1", "asnap_zerotrip" => "m/1", "asnap_outside" => "m/1", "asnap_nested" => "m/1",
        "asnap_ii" => "m/1", "asnap_nestedboth" => "m/1", "ttgc" => "mup/2",
        # `asnap_nameclash` snapshots only the array whose snapshot stack name is free
        "asnap_nameclash" => "m_snap/1",
    )
    for name in kernels
        want = get(expected, name, "-")
        got = plan_of(name)
        (got == want && plan_of(name; ii = true) == want) ||
            check(false, "plan $name: got $got (fuse_ii_loops: $(plan_of(name; ii = true))), want $want")
        want == "-" || check(got == want, "plan $name: $got")
    end
    with_plan = [n for n in kernels if plan_of(n) != "-"]
    check(sort(with_plan) == sort(collect(keys(expected))), "exactly $(length(expected)) corpus kernels get a snapshot: $(join(with_plan, " "))")
    for name in ("asnap_readbefore", "asnap_selfnl", "asnap_readbetween")
        check(plan_of(name) == "-", "plan $name: refused (a read of the array precedes a write to it)")
    end

    # 2. EFFECT
    (src, _) = kernel_of("ttgc")
    on = STADE.stade_adjoint(src; keep_push_pop = false, fuse_ii_loops = true)
    sa = string(on.adjoint)
    check(!occursin("mup_stack", sa) && occursin("mup_snap_stack", sa), "ttgc adjoint: the per-write stack of mup is gone, the snapshot stack is used")
    check(occursin("__len_mup", string(on.initstacks)) && occursin("length(mup)", string(on.initstacks)), "ttgc initstacks_*: reads length(mup) once")
    off = STADE.stade_adjoint(src; keep_push_pop = true)
    check(!occursin("snap", string(off.adjoint)) && !occursin("snap", string(off.initstacks)), "keep_push_pop = true: nothing changes")
    hv = STADE.stade_hvp(src; keep_push_pop = false, fuse_ii_loops = true)
    check(occursin("mup_snap_stack_d", string(hv.hvp)), "ttgc hvp: the snapshot stack has a shadow, as the adjoint's does")

    # 3. STATE
    function final_state(name, gen_adjoint, gen_init; kpp)
        (src, _) = kernel_of(name)
        kernel = STADE.parse_kernel(src)
        Random.seed!(sum(codeunits(name)))
        baseline = STADE.val_generate_baseline(kernel, src)
        fn = STADE.val_compile(gen_adjoint)
        ini = STADE.val_compile(gen_init)
        vals = deepcopy(baseline.values)
        Random.seed!(11)
        seed = STADE.val_random_values_like(kernel, baseline.values)
        names = Symbol[a for a in gen_init.args[1].args[2:end]]
        extra = [haskey(baseline.int_args, n) ? baseline.int_args[n] : deepcopy(vals[n]) for n in names]
        stacks = STADE.val_init_stacks(ini, extra)
        args = STADE.val_adjoint_call_args(kernel.sig, baseline.int_args, vals, seed, stacks)
        ret = Base.invokelatest(fn, args...)
        d = asnap_state(kernel, args, ret)
        return (d, baseline)
    end
    maxdiff = asnap_maxdiff
    function strip_restore(e)
        e isa Expr || return e
        args = Any[]
        for a in e.args
            if a isa Expr && a.head == :for && a.args[1] isa Expr && startswith(string(a.args[1].args[1]), "__sk_") &&
               any(s -> s isa Expr && s.head == :(=) && s.args[1] isa Expr && s.args[1].head == :ref && !endswith(string(s.args[1].args[1]), "_stack"),
                   a.args[2].args)
                continue
            end
            push!(args, strip_restore(a))
        end
        return Expr(e.head, args...)
    end
    for name in sort(collect(keys(expected)))
        (src, _) = kernel_of(name)
        for fuse in (false, true)
            ref = STADE.stade_adjoint(src; keep_push_pop = true, fuse_ii_loops = fuse)
            new = STADE.stade_adjoint(src; keep_push_pop = false, fuse_ii_loops = fuse)
            (sr, base) = final_state(name, ref.adjoint, ref.initstacks; kpp = true)
            (sn, _) = final_state(name, new.adjoint, new.initstacks; kpp = false)
            d = maxdiff(sr, sn)
            scale = asnap_scale(sr)
            check(d <= 1e-13 * scale, "state $name fuse=$fuse: arrays and adjoints equal the keep_push_pop=true result (max diff $(d))")
            fuse && (name in ("asnap_ok", "asnap_2d", "ttgc")) || continue
            sab = strip_restore(new.adjoint)
            check(sab != new.adjoint, "state $name: negative control removed a restore loop")
            (ss, _) = final_state(name, sab, new.initstacks; kpp = false)
            ds = maxdiff(sr, ss)
            check(ds > 1e-9 * scale, "state $name: negative control (no restore) changes the state by $(ds)")
        end
    end

    # 4. GUARDS on the plan itself
    (outer_empty, inner_empty) = asnap_synthetic_ii(joinpath(dir, "asnap_ok.jl"))
    check(outer_empty, "an ii-classified Jacobi loop gets no snapshot (G7)")
    check(inner_empty, "a write to the array inside an ii-classified loop of the body blocks the snapshot (G7)")

    # 5. SABOTAGE
    stade_src = read(joinpath(@__DIR__, "..", "src", "STADE.jl"), String)
    self = @__FILE__
    # `patches`: (old, new) pairs, each `old` occurring once. `probe` is the expression the fresh process prints after RESULT.
    function sabotaged_text(patches, probe)
        text = stade_src
        for (old, new) in patches
            count(old, text) == 1 || return "error: patch text not found exactly once: $(first(old, 50))"
            text = replace(text, old => new)
        end
        tmp = tempname() * ".jl"
        write(tmp, text)
        script = "include($(repr(tmp))); include($(repr(self))); println(\"RESULT \", $(probe))"
        out = try read(`$(Base.julia_cmd()) --startup-file=no -e $script`, String) catch e; "ERR" end
        rm(tmp; force = true)
        m = match(r"RESULT (.*)", out)
        return m === nothing ? "error: no result" : String(strip(m.captures[1]))
    end
    path(name) = joinpath(dir, name * ".jl")
    g4 = "agen_asnap_final_read_ok(B, A, kinds) || continue"
    g5 = "agen_asnap_count_refs(st.cond, A) > 0 && return true"
    g6 = "A ∉ handled && "
    g7_body = "agen_asnap_write_in_ii_loop(B, A, ii_plan, kinds) && continue"
    g7_loop = "ii_here = in_ii || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), nothing) !== nothing)"
    g9 = "block_here = in_block || agen_tier_b_walk(st.body, agen_collect_reassigned(st.body, true)) !== nothing"
    g10 = "agen_asnap_swept_write(B, A, false) || continue"
    g12 = "(haskey(kinds, Symbol(string(A) * \"_snap\")) || haskey(kinds, agen_asnap_len(A))) && continue"
    g13 = "agen_asnap_index_free(B, A, follow) || continue"
    g14 = "any(k -> get(sites, k, false) === true, agen_asnap_scatter_keys!(B, A, indirect, kinds, Any[])) || continue"
    restore_line = "append!(exprs, agen_asnap_restore(primal_body, kinds, ectx))"
    save_line = "exprs = agen_asnap_save(body, kinds, ectx)"

    # guards whose removal makes a wrong result: the kernel must fail an oracle or change the array state
    for (nm, patches, kernel) in (
            ("G4: a read before the last write is allowed", [(g4, "")], "asnap_readbefore"),
            ("G4: a read before the last write is allowed", [(g4, "")], "asnap_selfnl"),
            ("G4: a read before the last write is allowed", [(g4, "")], "asnap_readbetween"),
            ("the restore loop is not emitted", [(restore_line, "")], "asnap_ok"),
            ("the save loop is not emitted", [(save_line, "exprs = Any[]")], "asnap_ok"),
            ("G12: a snapshot stack may share a name with an array's stack", [(g12, "")], "asnap_nameclash"))
        r = sabotaged_text(patches, "sabotage_probe($(repr(path(kernel))))")
        # "loud": the generated code does not run at all, which is a failure nobody can miss
        check(r == "caught" || (r == "loud" && startswith(nm, "G12")), "sabotage \"$nm\": $kernel $(r == "caught" ? "fails an oracle or changes the array state" : r)")
    end
    # G9 refuses a ragged-block ancestor. Without it the layout reads a block-local offset as a global one.
    r = sabotaged_text([(g9, "block_here = false")], "sabotage_probe($(repr(path("asnap_ragged"))))")
    check(r in ("caught", "loud"), "sabotage \"G9: a ragged-block ancestor is allowed\": asnap_ragged $(r == "caught" ? "fails its oracle" : r)")

    # guards that only decide cost or an emission path: the plan must change, so the guard is not dead
    for (nm, patches, kernel, probe) in (
            ("G10: a body that writes one slot is snapshotted", [(g10, "")], "asnap_direct", "asnap_plan_string"),
            ("G13: a write that follows the loop variable is snapshotted", [(g13, "")], "asnap_slice", "asnap_plan_string"),
            ("G14: a loop with no scatter is snapshotted", [(g14, "")], "asnap_readbetween", "asnap_plan_string"),
            ("G6: the same array is snapshotted again in an inner loop", [(g6, "")], "asnap_nestedboth", "asnap_plan_string"))
        want = get(expected, kernel, "-")
        r = sabotaged_text(patches, "$(probe)($(repr(path(kernel))))")
        check(r != want, "sabotage \"$nm\": $kernel plan $want -> $r")
    end
    for (nm, patches) in (("G7: a write in an ii-classified loop is allowed", [(g7_body, "")]),
                          ("G7: an ii-classified loop may hold a snapshot", [(g7_loop, "ii_here = false")]))
        r = sabotaged_text(patches, "asnap_synthetic_ii($(repr(path("asnap_ok"))))")
        check(r != "(true, true)", "sabotage \"$nm\": synthetic ii plan gives $r")
    end
    # Reported, not required: the condition rule is cautious. The reverse sweep replays a branch from its saved flag, so no oracle can tell.
    r = sabotaged_text([(g5, "")], "asnap_plan_string($(repr(path("asnap_condread"))))")
    println(rpad("info", 6), "sabotage \"G5: a read in an if condition is allowed\": asnap_condread plan - -> $r (cautious guard, no oracle can fail)")
    println("\n", bad == 0 ? "all array snapshot checks passed" : "*** $bad array snapshot check(s) failed ***")
    return bad
end

# One kernel through the three gradient oracles and the state comparison, with `keep_push_pop = false` and both fuse settings. Returns "caught" when
# any of them fails, "loud" when the generator or the compiler throws, "missed" when everything passes.
function sabotage_probe(path::String)
    name = splitext(basename(path))[1]
    caught = false
    for fuse in (true, false)
        try
            a = STADE.stade_validate_adjoint_file(path; trials = 4, keep_push_pop = false, fuse_ii_loops = fuse)
            d = STADE.stade_validate_dotprod_file(path; trials = 4, keep_push_pop = false, fuse_ii_loops = fuse)
            h = STADE.stade_validate_hvp_file(path; trials = 4, keep_push_pop = false, fuse_ii_loops = fuse)
            caught |= !(a.ok && d.ok && h.ok)
            src = STADE.io_read_corpus_entry(path)
            ref = STADE.stade_adjoint(src; keep_push_pop = true, fuse_ii_loops = fuse)
            new = STADE.stade_adjoint(src; keep_push_pop = false, fuse_ii_loops = fuse)
            kernel = STADE.parse_kernel(src)
            states = map(((ref.adjoint, ref.initstacks), (new.adjoint, new.initstacks))) do (adj, ini)
                Random.seed!(sum(codeunits(name)))
                baseline = STADE.val_generate_baseline(kernel, src)
                vals = deepcopy(baseline.values)
                Random.seed!(11)
                seed = STADE.val_random_values_like(kernel, baseline.values)
                names = Symbol[x for x in ini.args[1].args[2:end]]
                extra = [haskey(baseline.int_args, n) ? baseline.int_args[n] : deepcopy(vals[n]) for n in names]
                stacks = STADE.val_init_stacks(STADE.val_compile(ini), extra)
                args = STADE.val_adjoint_call_args(kernel.sig, baseline.int_args, vals, seed, stacks)
                ret = Base.invokelatest(STADE.val_compile(adj), args...)
                return asnap_state(kernel, args, ret)
            end
            caught |= asnap_maxdiff(states[1], states[2]) > 1e-9 * asnap_scale(states[1])
        catch e
            println(stderr, "sabotage_probe($(name)): ", first(sprint(showerror, e), 300))
            return "loud"
        end
    end
    return caught ? "caught" : "missed"
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_array_snapshot() == 0 ? 0 : 1)
end
