# hoist_nested(y, x, i_n, i_nb)
#
# Two sequential loops inside one thread, both invariant for the target: row
# i_r adds into y[i_b] from inside a middle loop of 2 iterations and an inner
# ragged loop. The inner flush lands in the middle loop, where it is itself an
# invariant accumulation and hoists again, so the row issues one atomic.
#
# y: bin totals, length i_nb, accumulated in place
# x: input array of length 8 * i_n
# i_n: number of rows
# i_nb: number of bins
function hoist_nested(y, x, i_n, i_nb)
    for i_r = 1:i_n
        i_b = mod(i_r - 1, i_nb) + 1
        for i_m = 1:2
            for i_e = 1:mod(i_r, 3) + 1
                y[i_b] = y[i_b] + x[(i_r - 1) * 8 + (i_m - 1) * 4 + i_e]
            end
        end
    end
    return nothing
end
