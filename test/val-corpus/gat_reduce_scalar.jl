# gat_reduce_scalar(x, y, s, n)
#
# A stencil loop that also multiplies a scalar over its iterations. The adjoint of the running product is carried from one iteration to the next, so the rewrite must refuse the loop.
#
# x: input, length n
# y: output, length n, interior filled in place
# s: length-1 output, running product of the interior outputs
# n: number of points
function gat_reduce_scalar(x, y, s, n)
    y[1] = x[1]
    y[n] = x[n]
    t = 1.0
    # interior with a running product
    for i = 2:n - 1
        y[i] = x[i - 1] + 2.0 * x[i + 1]
        t = t * (1.0 + 0.1 * y[i])
    end
    s[1] = t
    return nothing
end
