# so_hostread(x, w, y, q, loss, i_n, i_nc)
#
# Like so_tile, but `wv` is also READ outside every loop that assigns it: `q[1] = wv * wv` consumes the value `wv` held before
# the tile loop. The reverse sweep restores that value past the tile loop from the entry save of the first tap, which a split
# thread would have filled with a placeholder. The tile loop must stay on the host.
#
# x: input, length i_n + 2
# w: weights, length 3 * i_nc
# y: output, length i_n
# q: length-1 array
# loss: length-1 output array, accumulated in place
# i_n, i_nc: trip counts
function so_hostread(x, w, y, q, loss, i_n, i_nc)
    wv = w[1] * 2.0
    q[1] = wv * wv
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
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k] + q[1]
    end
    return nothing
end
