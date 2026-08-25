# coarsen_retire.jl
#
# A ragged (Tier B) kernel whose level bound is reassigned AFTER the
# loop it governs, rather than before it.
#
# Every other ragged kernel in this corpus writes its bound at the TOP
# of the level body (mg_vcycle's `n = nl - 1`), which left the
# trailing-reassignment case unreachable -- and it was wrong:
# agen_tier_b_block_stmts recorded each level's val_*/size table from
# the value AFTER the body's reassignment, i.e. the NEXT level's, so
# every block was sized one coarsening short and overran into its
# successor. Wrong gradients in :indexed mode only; :stack mode
# computes no sizes at all and validated clean throughout.
#
# The halving rounds UP so no level is ever degenerate (cur never
# reaches zero) for any integer draw the baseline generator makes.
#
# x: input field, length >= n
# y: accumulator, length >= n, updated in place
# n: finest level width
# levels: number of coarsening passes
# out: single-element output
function coarsen_retire(x, y, n, levels, out)
    cur = n
    for i_l = 1:levels
        for i = 1:cur
            t = x[i] * x[i]
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
    return nothing
end