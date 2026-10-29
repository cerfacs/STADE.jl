# gat_accum_out(x, y, n)
#
# The primal accumulates into y instead of overwriting it, so the adjoint reads yb[i] without a reset.
#
# x: input, length n
# y: accumulator, length n, input and output
# n: number of points
function gat_accum_out(x, y, n)
    # interior accumulation from both neighbours
    for i = 2:n - 1
        y[i] = y[i] + 0.5 * (x[i - 1] + x[i + 1])
    end
    return nothing
end
