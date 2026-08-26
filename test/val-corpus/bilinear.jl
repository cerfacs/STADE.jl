# bilinear(loss, x, a, y, i_m, i_n)
#
# Bilinear form x' * A * y, accumulated into a length-1 output array.
# The i_m x i_n nest is rectangular and every iteration reads and writes
# only at indices derived from its own flattened index, so rule 2 applies
# and it is written as one loop over 1:i_m * i_n. The accumulation into
# loss[1] is a commutative reduction at a FIXED index, not a value
# carried between iterations: cgen_reduction_only_loop admits exactly
# this shape as an atomic add, so it does not block the fusion the way a
# genuine recurrence (u[i] = c * u[i - 1]) would.
#
# loss: length-1 output array, accumulated in place
# x: left vector of length i_m
# a: matrix of shape (i_m, i_n)
# y: right vector of length i_n
# i_m: number of rows of a
# i_n: number of columns of a
function bilinear(loss, x, a, y, i_m, i_n)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
    end
    return nothing
end
