# gat_two_targets(a, x, y, z, n)
#
# One loop writes two outputs that read the same inputs at different offsets, so the adjoint has two gathered arrays with different offset sets.
#
# a: coefficients, length n, differentiated
# x: input, length n
# y: first output, length n, interior filled in place
# z: second output, length n, interior filled in place
# n: number of points
function gat_two_targets(a, x, y, z, n)
    y[1] = x[1]
    y[n] = x[n]
    z[1] = a[1]
    z[n] = a[n]
    # interior: y uses a at the left neighbour, z uses x at the right neighbour
    for i = 2:n - 1
        y[i] = a[i - 1] * x[i]
        z[i] = x[i + 1] * a[i] + y[i]
    end
    return nothing
end
