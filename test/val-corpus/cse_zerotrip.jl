# cse_zerotrip(out, u, w, i_npass, i_w0)
#
# Coverage kernel for a temporary whose run straddles a loop that may not
# execute.
#
# The same subexpression is computed before and after an inner loop whose
# width retires to zero. emit_cse_block ends the run at the `for`, so the
# two occurrences land in different runs and are named separately -- a
# temporary can never be carried across a loop header, which is what would
# break when the loop runs zero times and the carried name is stale (or
# when it runs and the loop body writes something the expression reads).
#
# The inner loop writes `s`, which the expression after it reads, so
# merging across the loop would be wrong even on a pass where it does run.
# Retiring by three per pass reaches a non-positive width on every draw,
# the same construction entry_empty uses.
#
# out: output array of length i_npass, filled in place
# u: input array of length >= i_w0
# w: input array of length >= i_w0
# i_npass: number of passes
# i_w0: initial window width
function cse_zerotrip(out, u, w, i_npass, i_w0)
    s = u[1] * w[1]
    width = i_w0
    for i_p = 1:i_npass
        # before the inner loop
        s = s + u[1] * w[1] * u[1]
        for i_j = 1:width
            s = s + u[i_j] * w[i_j]
        end
        # after it: same subexpression, but `s` may or may not have moved
        out[i_p] = s * (u[1] * w[1])
        width = width - 3
    end
    return nothing
end
