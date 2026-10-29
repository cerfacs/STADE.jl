# hoist_bins(y, z, x, w, v, i_n, i_nb)
#
# Per-row ragged loop whose accumulation target does not depend on the loop.
# Row i_r owns bin i_b and runs mod(i_r, 4) + 1 inner iterations, so the nest
# is not rectangular and only the outer loop becomes threads. Every inner
# iteration adds into y[i_b], one element that all rows with the same bin
# share, and reads v[i_b]. The adjoint adds into v's shadow at i_b on every
# inner iteration (an invariant index, hoistable) and into w's shadow at the
# inner index (not invariant, stays one atomic per iteration).
#
# y: bin totals, length i_nb, accumulated in place
# z: per-slot output, length 4 * i_n, filled in place
# x: input array of length 4 * i_n
# w: weight indexed by the inner iteration, length 4
# v: weight indexed by the bin, length i_nb
# i_n: number of rows
# i_nb: number of bins
function hoist_bins(y, z, x, w, v, i_n, i_nb)
    for i_r = 1:i_n
        # bin of this row; fixed for the whole inner loop
        i_b = mod(i_r - 1, i_nb) + 1
        for i_e = 1:mod(i_r, 4) + 1
            i_s = (i_r - 1) * 4 + i_e
            y[i_b] = y[i_b] + x[i_s] * w[i_e]
            z[i_s] = x[i_s] * v[i_b]
        end
    end
    return nothing
end
