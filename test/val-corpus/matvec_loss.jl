# matvec_loss(loss, a, u, v, i_m, i_n)
#
# Matrix-vector product v = A * u followed by the squared-norm loss of v.
# The i_m x i_n product nest is rectangular and independent, so rule 2
# collapses it to one loop over 1:i_m * i_n; the accumulation into
# v[i_i] is a scatter-accumulate at an index derived from the flattened
# index, which cgen_device_assign turns into an atomic add. The loss loop
# is already flat and is a fixed-index reduction, so it stays as it is --
# it is not part of the nest and has a different trip count.
#
# loss: length-1 output array, accumulated in place
# a: matrix of shape (i_m, i_n)
# u: input vector of length i_n
# v: output vector of length i_m, caller-zeroed, accumulated in place
# i_m: number of rows of a
# i_n: number of columns of a
function matvec_loss(loss, a, u, v, i_m, i_n)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        v[i_i] = v[i_i] + a[i_i, i_j] * u[i_j]
    end
    for i_i2 = 1:i_m
        loss[1] = loss[1] + v[i_i2] ^ 2
    end
    return nothing
end
