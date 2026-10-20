# cross_ifwrite(x, w, z, loss, i_n, i_nc)
#
# A second, conditional write of the scalar after its first one: the value at the end of an iteration is the pure load or its doubling,
# depending on data, so the last top-level write does not define it.
#
# x: length i_n + i_nc, w: length i_nc, z: length i_n, loss: length 1
function cross_ifwrite(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[t + ci - 1]
            a = a + xv * xv * w[ci]
            if x[t + ci] > 0.5
                xv = xv * 2.0
            end
            a = a + xv * xv
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
