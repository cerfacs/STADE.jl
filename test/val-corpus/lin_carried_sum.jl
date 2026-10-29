# lin_carried_sum(x, y, z, n)
#
# A scalar carried from one iteration to the next: read in a sum at the top of the body, then assigned in both arms of a branch. The entry value of s
# is live (it reaches y), so no reset of s at the top of the body is allowed. s is also read in a product, so its value is needed.
#
# x: input, length n
# y: output, length n, receives the carried value plus one
# z: output, length n, receives the square of the new carried value
# n: number of elements
function lin_carried_sum(x, y, z, n)
    s = 1.0
    for i = 1:n
        # linear read of the carried value
        y[i] = s + 1.0
        if x[i] > 0.0
            s = x[i]
        else
            s = 2.0 * x[i]
        end
        z[i] = s * s
    end
    return nothing
end
