# gat_nonlinear(x, y, n)
#
# Nonlinear terms: the adjoint's scalars (cosines, products) are recomputed at the shifted index when gathered.
#
# x: input, length n
# y: output, length n, interior filled in place
# n: number of points
function gat_nonlinear(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    # interior
    for i = 2:n - 1
        y[i] = sin(x[i - 1]) * x[i + 1] + exp(0.1 * x[i])
    end
    return nothing
end
