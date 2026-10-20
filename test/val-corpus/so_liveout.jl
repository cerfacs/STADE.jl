# so_liveout(x, w, y, z, loss, i_n, i_nc)
#
# A batch loop whose iteration assigns `wv` in an inner tile loop and then READS it after that loop (`z[b] = wv * ...`): the
# value read is the last tile's final tap weight, which a split tile loop would leave unwritten on the host. The inner tile loop
# must stay on the host; the batch loop reads `wv` only after assigning it, so it has no such problem by itself.
#
# x: input, length i_n + 2
# w: weights, length 3 * i_nc
# y: output, length i_n
# z: output, length i_nb
# loss: length-1 output array, accumulated in place
# i_nb, i_n, i_nc: trip counts
function so_liveout(x, w, y, z, loss, i_nb, i_n, i_nc)
    for bb = 1:i_nb
        wv = 0.0
        for t = 1:i_n
            a = 0.0
            for ci = 1:i_nc
                wv = w[(ci - 1) * 3 + 1]
                a = a + x[t] * wv
                wv = w[(ci - 1) * 3 + 2]
                a = a + x[t + 1] * wv
                wv = w[(ci - 1) * 3 + 3]
                a = a + x[t + 2] * wv
            end
            y[t] = a
        end
        z[bb] = wv * 3.0
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
    for k = 1:i_nb
        loss[1] = loss[1] + z[k] * z[k]
    end
    return nothing
end
