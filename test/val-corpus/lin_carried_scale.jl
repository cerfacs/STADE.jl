# lin_carried_scale(x, y, z, n)
#
# The same carried scalar, read through a literal product at the top of the body. The entry value of s is live, so the body must keep it.
#
# x: input, length n
# y: output, length n, receives half the carried value
# z: output, length n, receives the square of the new carried value
# n: number of elements
function lin_carried_scale(x, y, z, n)
    s = 1.0
    for i = 1:n
        # literal scaling of the carried value
        y[i] = 0.5 * s
        if x[i] > 0.0
            s = x[i]
        else
            s = 2.0 * x[i]
        end
        z[i] = s * s
    end
    return nothing
end
