# cross_branchlocal(x, w, z, loss, i_n, i_nc)
#
# The scalar is written inside an `if` of the first loop and at the top of the second. Every read follows a write in its own iteration, but the
# first loop's value is branch-local: the reverse sweep recovers it through the restores of whatever writes the scalar next, which are exactly
# the first-write saves the second loop's transformation drops. One loop transformed and the other not is wrong, so the scalar is refused.
#
# x: length i_n + i_nc, w: length i_nc, z: length 2 * i_n, loss: length 1
function cross_branchlocal(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            if x[t + ci - 1] > 0.5
                xv = x[t + ci - 1] * 2.0
                a = a + xv * xv * w[ci]
            end
        end
        z[t] = a
    end
    for t = 1:i_n
        b = 0.0
        for ci = 1:i_nc
            xv = x[t + ci - 1]
            b = b + xv * xv * w[ci]
        end
        z[i_n + t] = b
    end
    for m = 1:2 * i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
