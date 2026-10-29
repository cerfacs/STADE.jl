# gat_wide(x, w, y, n)
#
# Five-point stencil with a coefficient array and offsets -2 to +2. Both x and w receive a scatter in the adjoint, at different offsets sets.
#
# x: input, length n
# w: weights, length n, differentiated
# y: output, length n, interior filled in place
# n: number of points
function gat_wide(x, w, y, n)
    y[1] = x[1]
    y[2] = x[2]
    y[n - 1] = x[n - 1]
    y[n] = x[n]
    # interior five-point weighted stencil
    for i = 3:n - 2
        y[i] = w[i] * x[i - 2] + x[i - 1] * x[i + 1] + w[i + 1] * x[i + 2] - 2.0 * x[i]
    end
    return nothing
end
