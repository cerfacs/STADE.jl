# gat_sum_scalar(x, y, s, n)
#
# A stencil loop that also sums a scalar over its iterations. The adjoint of a running sum is the same value in every iteration, so the backward loop is still a gather candidate.
#
# x: input, length n
# y: output, length n, interior filled in place
# s: length-1 output, running sum of the interior outputs
# n: number of points
function gat_sum_scalar(x, y, s, n)
    y[1] = x[1]
    y[n] = x[n]
    t = 0.0
    # interior with a running sum
    for i = 2:n - 1
        y[i] = x[i - 1] + 2.0 * x[i + 1]
        t = t + y[i] * y[i]
    end
    s[1] = t
    return nothing
end
