# retire_empty(u, x, i_npass, i_w0)
#
# Retires the active window by three points per pass, deliberately past
# zero: with the baseline generator's integer range the third pass always
# reaches a NEGATIVE bound.
#
# Guard kernel for stack sizing. `for i_j = 1:w` with w < 0 runs zero
# times, but the trip-count formula div(w - 1, 1) + 1 evaluates to w, so a
# size sum that adds it up is short by |w| and the reverse sweep indexes
# off the end of the stack. Only the indexed mode (keep_push_pop = false)
# is affected -- growable :stack mode never needs a size -- which is why
# the CPU corpus never saw it. mg_vcycle carries the same shape at
# num_levels >= 4 and was silently unreachable because its committed
# baseline happens to hold num_levels = 3; this kernel makes the case
# deterministic instead of leaving it to a draw.
#
# Caller responsibility: u and x must be at least i_w0 long. The window
# going negative is the point, not an oversight.
#
# u: array of length >= i_w0, updated in place
# x: coefficient array of the same length
# i_npass: number of passes
# i_w0: initial window width
function retire_empty(u, x, i_npass, i_w0)
    w = i_w0
    for i_p = 1:i_npass
        # nonlinear in u, so the write needs a snapshot and a real stack
        for i_j = 1:w
            u[i_j] = u[i_j] + x[i_j] * u[i_j]
        end
        # retires past zero on the third pass for every baseline draw
        w = w - 3
    end
    return nothing
end
