# cross_readbefore(x, w, z, loss, i_n, i_nc)
#
# The scalar is read BEFORE its first write in the iteration, so it carries the previous iteration's value into this one: the save at the
# first write cannot be dropped, whatever the rhs of the write looks like. A pure rhs makes this a trap for the cross-iteration remat.
#
# x: length i_n + i_nc, w: length i_nc, z: length i_n, loss: length 1
function cross_readbefore(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        xv = x[t] * 0.5
        for ci = 1:i_nc
            a = a + xv * xv * w[ci]
            xv = x[t + ci - 1]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
