include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_liveout(dir="val-corpus")

Regression gate for the live-out rule of the `fuse_ii_loops` classifier (`snap_ii_classify`/`agen_ii_classify`).

`:independent` fuses a loop's adjoint into the FORWARD position, so it may only consume adjoints that already exist there.
The adjoint of an active scalar the loop assigns and later code reads is seeded by that code's backward sweep, which runs
later. Two shapes break it:

  liveout_red_nest   an accumulator read after its loop, the enclosing loop not fused   -> the loop is `:mixed`
  liveout_red_fused  the same with the enclosing loop also fused                         -> inner `:mixed`
  liveout_red_top    top-level accumulator, value-needed read after the loop             -> `:mixed` (already so; shape guard)
  liveout_local_last a plain local overwritten every iteration and read after the loop   -> not covered at all

For the last, an un-reversed fused loop would hand the adjoint of the final value to the FIRST iteration.

Two groups of checks:

  1. PLAN: the classification of each loop of the four kernels above.
  2. SABOTAGE: in a fresh process each half of the rule is removed from both classifier copies; the kernel that depends on
     it must then fail its adjoint or dot-product oracle (`keep_push_pop = false`, `fuse_ii_loops = true`).
"""
function validate_liveout(dir::String = joinpath(@__DIR__, "val-corpus"))
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))

    function plan_of(name)
        src = Meta.parse(read(joinpath(dir, name * ".jl"), String))
        kernel = STADE.norm_insert_dead_entry_resets(STADE.parse_kernel(src))
        plan = STADE.stade_ii_plan_check(kernel)
        return sort([(k[2], String(v)) for (k, v) in plan])
    end

    expected = Dict(
        "liveout_red_nest" => [(1, "recompute"), (2, "mixed")],
        "liveout_red_fused" => [(1, "independent"), (2, "mixed")],
        "liveout_red_top" => [(2, "mixed")],
        "liveout_local_last" => Tuple{Int,String}[],
    )
    for (name, want) in sort(collect(expected))
        got = plan_of(name)
        check(got == want, "plan $name: $got $(got == want ? "" : "want $want")")
    end

    src = read(joinpath(@__DIR__, "..", "src", "STADE.jl"), String)
    function sabotaged(a, b, kernel)
        count(a, src) == 2 || return :error
        tmp = tempname() * ".jl"
        write(tmp, replace(src, a => b))
        script = """
            include($(repr(tmp)))
            path = $(repr(joinpath(dir, kernel * ".jl")))
            a = STADE.stade_validate_adjoint_file(path; trials = 4, keep_push_pop = false, fuse_ii_loops = true)
            d = STADE.stade_validate_dotprod_file(path; trials = 4, keep_push_pop = false, fuse_ii_loops = true)
            println("RESULT ", a.ok, " ", d.ok)
        """
        out = try read(`$(Base.julia_cmd()) --startup-file=no -e $script`, String) catch e; "ERR" end
        rm(tmp; force = true)
        m = match(r"RESULT (true|false) (true|false)", out)
        m === nothing && return :error
        return (m.captures[1] == "false" || m.captures[2] == "false") ? :caught : :missed
    end
    sab = [
        ("an accumulator is never deferred", "defer = !isempty(live_out)", "defer = false", "liveout_red_nest"),
        ("an accumulator is never deferred", "defer = !isempty(live_out)", "defer = false", "liveout_red_fused"),
        ("a live-out plain local is fused", "isempty(setdiff(live_out, redvars)) || return :none", "", "liveout_local_last"),
    ]
    for (nm, a, b, kernel) in sab
        r = sabotaged(a, b, kernel)
        check(r == :caught, "sabotage \"$nm\": $kernel $(r == :caught ? "fails its oracle" : String(r))")
    end
    println("\n", bad == 0 ? "all live-out checks passed" : "*** $bad live-out check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_liveout() == 0 ? 0 : 1)
end
