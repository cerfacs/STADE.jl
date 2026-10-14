# idm_zerotrip(loss, y, x, w, i_n)
#
# Same body as idm_slots, but the loop bound is computed inside the kernel
# and retired past zero. Every draw of i_n in 3:5 gives zero, one or two
# iterations, so the slot loop runs zero times on some draws. A loop that
# may not run must not leave any claim behind about what it wrote.
#
# loss: length-1 output array, accumulated in place
# y: output array of length 4 * i_n
# x: input array of length i_n + 1
# w: weight array of length 3
# i_n: number of points
function idm_zerotrip(loss, y, x, w, i_n)
    i_m = i_n - 3
    for i_t = 1:i_m
        i_b = 4 * (i_t - 1)
        y[i_b + 1] = x[i_t] * w[1] + x[i_t + 1] * w[2]
        y[i_b + 2] = x[i_t] * w[2] + x[i_t + 1] * w[3]
        y[i_b + 3] = x[i_t] * w[3] + x[i_t + 1] * w[1]
        y[i_b + 4] = x[i_t] * w[1] + x[i_t + 1] * w[3]
    end
    # y is read over its full length, so the unwritten slots enter the loss
    for i_k = 1:4 * i_n
        loss[1] = loss[1] + y[i_k] * y[i_k]
    end
    return nothing
end
