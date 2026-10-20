# cross_after(x, w, z, loss, i_n, i_nc)
#
# The scalar is read AFTER its loop, in the same outer iteration: the read sees the last iteration's value, and in the reverse sweep the
# outer iteration begins there. Rebuilding the end-of-iteration value at the head of every reverse inner iteration does not provide it.
#
# x: length i_n + i_nc, w: length i_nc, z: length i_n, loss: length 1
function cross_after(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        xv = 0.0
        for ci = 1:i_nc
            xv = x[t + ci - 1]
            a = a + xv * xv * w[ci]
        end
        a = a + xv * xv
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
