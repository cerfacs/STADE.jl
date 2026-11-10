# Pins a known limit (defect B6 of the offload battle test, not a defect of the gate).
# An independent loop that rewrites a local scalar (`t1 = x[i1]^2; t1 = t1 + 1.0`) has a snapshot stack for `t1` when the adjoint or the HVP
# is generated with `keep_push_pop = false` and `fuse_ii_loops = false`: the forward loop saves the t1 of the previous iteration and the reverse
# loop reads the t1 restored by the next one, so both loops carry a value between iterations and must stay on the host (correct, slower).
# With `fuse_ii_loops = true` the stack is proven unnecessary and the loop is offloaded.
# This test fails if either side changes: a false offload of the first case would give wrong gradients; a change of the second means the limit moved
# and the README and claude/skill-stade-dev.md text must follow.
include(joinpath(@__DIR__, "validate_battle_status.jl"))

function validate_scalar_stack_limit(; verbose = true)
    bad = 0; checks = 0
    for r in bt_status_rows("f1_scalar_priv.jl"; modes = (:adjoint, :hvp))
        get(r, :kind, :none) === :status || continue
        checks += 1
        want = r.fuse ? Dict(:K => 1) : Dict(:H => 2)
        if r.observed != want
            bad += 1
            verbose && println("NOT OK ", r.mode, " ", r.backend, " fuse_ii_loops=", r.fuse, ": observed ", r.observed, ", expected ", want)
        end
    end
    checks == 8 || (bad += 1; verbose && println("NOT OK expected 8 status rows, got ", checks))
    println(bad == 0 ? "all $checks scalar-stack limit checks passed" : "$bad of $checks scalar-stack limit checks NOT OK")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_scalar_stack_limit() || exit(1)
end
