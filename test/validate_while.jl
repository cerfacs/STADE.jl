include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

# `while` kernels live in their own directory rather than in val-corpus,
# because they need keep_push_pop = true and the main corpus runs both
# storage modes. Without this script nothing would run them: the feature
# would pass on the day it was written and rot quietly afterwards.
#
# Three things are checked here, not one. The four oracles on every
# positive kernel, the two parse refusals, and the keep_push_pop = false
# refusal. A refusal with no test is a refusal that can silently turn into
# an acceptance.

const WHILE_DIR = joinpath(@__DIR__, "while-corpus")

"""
    validate_while()

Run the four oracles over the `while` corpus in the supported storage
mode, then check every refusal fires with its own message.
"""
function validate_while(; trials::Int = 3, int_lo::Int = 0, int_hi::Int = 4)
    bad = 0

    # Driven directly rather than through validate_corpus, which runs the
    # main corpus at include time. Same four oracles, same entry points.
    subjects = sort([splitext(f)[1] for f in readdir(WHILE_DIR)
                     if endswith(f, ".jl") && !occursin(r"_(b|d|hv)\.jl$", f) &&
                        !startswith(f, "while_assign_cond") && !startswith(f, "while_no_progress")])
    for fuse in (false, true)
        println("--- oracles, fuse_ii_loops = ", fuse)
        for name in subjects
            path = joinpath(WHILE_DIR, name * ".jl")
            Random.seed!(hash(name))
            for (mode, gen, suffix, val) in (
                    (:tangent, STADE.stade_tangent_file, "_d.jl", STADE.stade_validate_tangent_file),
                    (:adjoint, STADE.stade_adjoint_file, "_b.jl", STADE.stade_validate_adjoint_file),
                    (:hvp,     STADE.stade_hvp_file,     "_hv.jl", STADE.stade_validate_hvp_file))
                try
                    gen(path, joinpath(WHILE_DIR, name * suffix);
                        keep_push_pop = true, fuse_ii_loops = fuse)
                    r = val(path; trials = trials, keep_push_pop = true, fuse_ii_loops = fuse,
                            int_lo = int_lo, int_hi = int_hi)
                    r.ok || (bad += 1)
                    println(rpad("$name [$mode]", 34), r.ok ? "ok  " : "FAIL  ",
                            round(r.max_rel_err, sigdigits = 4))
                catch e
                    bad += 1
                    println(rpad("$name [$mode]", 34), "ERR  ", first(sprint(showerror, e), 90))
                end
            end
            try
                r = STADE.stade_validate_dotprod_file(path; trials = trials,
                        keep_push_pop = true, fuse_ii_loops = fuse)
                r.ok || (bad += 1)
                println(rpad("$name [dotprod]", 34), r.ok ? "ok  " : "FAIL  ",
                        round(r.max_rel_err, sigdigits = 4))
            catch e
                bad += 1
                println(rpad("$name [dotprod]", 34), "ERR  ", first(sprint(showerror, e), 90))
            end
        end
    end

    println("\n--- refusals")
    checks = 0
    function expect_refusal(label, path, fragment, f)
        checks += 1
        msg = try
            f(path); ""
        catch e
            sprint(showerror, e)
        end
        if isempty(msg)
            bad += 1
            println(rpad(label, 46), " FAIL  accepted")
        elseif !occursin(fragment, msg)
            bad += 1
            println(rpad(label, 46), " FAIL  wrong reason: ", first(msg, 90))
        else
            println(rpad(label, 46), " ok")
        end
    end

    # A condition is an expression. The forward sweep evaluates it once per
    # iteration and the backward sweep never evaluates it, so a side effect
    # there would happen a different number of times in each.
    expect_refusal("condition that assigns", joinpath(WHILE_DIR, "while_assign_cond.jl"),
                   "assigns", p -> STADE.parse_kernel(STADE.io_read_corpus_entry(p)))

    # STADE does not prove termination. It checks only that the condition
    # reads something the body writes, which catches the loop that plainly
    # cannot end.
    expect_refusal("condition the body cannot change", joinpath(WHILE_DIR, "while_no_progress.jl"),
                   "cannot end", p -> STADE.parse_kernel(STADE.io_read_corpus_entry(p)))

    # keep_push_pop = false sizes every stack in closed form, and a while
    # has no trip count until it has run. This is what confines a while to
    # the CPU path, so it is the refusal most worth pinning down.
    expect_refusal("keep_push_pop = false", joinpath(WHILE_DIR, "while_count.jl"),
                   "no trip count",
                   p -> STADE.stade_adjoint(STADE.io_read_corpus_entry(p); keep_push_pop = false))

    # A while must never reach a device kernel body. cgen_body keeps the
    # loop on the host; this pins the guard behind that decision.
    checks += 1
    msg = try
        STADE.cgen_device_body(NamedTuple[(kind = :while, cond = :(i < n), body = NamedTuple[])],
                               :__tid, STADE.cgen_backend_cuda(), Set{Symbol}(),
                               Set{Symbol}(), Set{Symbol}())
        ""
    catch e
        sprint(showerror, e)
    end
    if occursin("cannot run inside a device kernel", msg)
        println(rpad("while refused inside a device kernel", 46), " ok")
    else
        bad += 1
        println(rpad("while refused inside a device kernel", 46), " FAIL  ", first(msg, 80))
    end

    println("\n", checks, " refusal checks run",
            bad == 0 ? "   all while checks passed" : "   *** $bad NOT OK ***")
    return bad
end

validate_while()
