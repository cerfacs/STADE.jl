# so_carry(x, w, y, loss, i_n, i_nc)
#
# A tile loop carried by a running scalar (`run = run * 0.5 + a`), so the tile loop can never be split, while the tap loop inside
# it assigns `wv` three times per tap block like so_tile. Only the tap loop is a candidate. Every `wv` is a pure load, so the
# cross-iteration remat drops the first save of each block and rebuilds the block's last weight at the head of each reverse
# iteration. What is left in the forward tap loop is the end-of-body restore of the tile loop, which reads `wv` after it: the
# forward tap loop stays on the host (live-out), and the reverse tap loop, which no longer needs the carried value, offloads.
# so_inner is the control for the save-only push guard, because its second write cannot be rebuilt.
#
# x: input, length i_n + 2
# w: weights, length 3 * i_nc
# y: output, length i_n
# loss: length-1 output array, accumulated in place
# i_n, i_nc: trip counts
function so_carry(x, w, y, loss, i_n, i_nc)
    run = 0.0
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
        run = run * 0.5 + a
        y[t] = run
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
    return nothing
end
