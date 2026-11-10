# Regression for the accumulate-and-read gate (defects B1 and B2 of the offload battle test).
# A loop that accumulates into an array and also reads that array in another place is order-dependent
# unless every site addresses the array through the loop variable.  Such a loop stays on the host.
# Each subject below must show no false offload, on both backends, in the primal.
include(joinpath(@__DIR__, "validate_battle_status.jl"))

function validate_acc_read_gate(; verbose = true)
    bad = 0; checks = 0
    for file in ("f4_seq_inner_ind.jl", "f9_scatter_rw.jl")
        for r in bt_status_rows(file; modes = (:primal,), fuses = (true,))
            get(r, :kind, :none) === :status || continue
            checks += 1
            v = bt_verdict(r.expected, r.observed, r.backend)
            if v === :false_offload
                bad += 1
                verbose && println("NOT OK ", file, " ", r.backend, " loop ", r.loop, ": offloaded, expected ", r.expected)
            end
        end
    end
    println(bad == 0 ? "all $checks accumulate-and-read gate checks passed" : "$bad of $checks accumulate-and-read gate checks NOT OK")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_acc_read_gate() || exit(1)
end
