# hoist_sentinel(y, x, w, i_n)
#
# The bin index is 0 exactly when the inner loop does not run: row i_r has
# i_b = mod(i_r, 3) inner iterations and accumulates into y[i_b]. A hoisted
# accumulation must therefore not flush after a loop that ran zero times,
# since y[0] does not exist. On the CPU the empty loop never touches y.
#
# y: bin totals, length 2, accumulated in place
# x: input array of length 2 * i_n
# w: weight indexed by the inner iteration, length 2
# i_n: number of rows
function hoist_sentinel(y, x, w, i_n)
    for i_r = 1:i_n
        i_b = mod(i_r, 3)
        for i_e = 1:i_b
            y[i_b] = y[i_b] + x[(i_r - 1) * 2 + i_e] * w[i_e]
        end
    end
    return nothing
end
