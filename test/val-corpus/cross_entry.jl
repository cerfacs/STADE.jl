# cross_entry(x, w, z, loss, i_n, i_nc)
#
# The scalar has a live ENTRY value: it is read before the loop and then overwritten by the loop's first write, so the save at that write
# is what brings the entry value back for the earlier read. Nothing inside the loop reads it before the write, which is exactly why only a
# kernel-wide read count can see it.
#
# x: length i_n + i_nc, w: length i_nc, z: length i_n, loss: length 1
function cross_entry(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        xv = x[t] * 0.5
        a = a + xv * xv
        for ci = 1:i_nc
            xv = x[t + ci - 1]
            a = a + xv * xv * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
    return nothing
end
