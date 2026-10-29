# gat_diff1(x, y, n)
#
# Forward difference. The adjoint scatters into xb at offsets 0 and +1 only, so the gather range is not centred on the loop range.
#
# x: input, length n
# y: output, length n, interior filled in place
# n: number of points
function gat_diff1(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    # interior forward difference
    for i = 2:n - 1
        y[i] = x[i + 1] - x[i]
    end
    return nothing
end
