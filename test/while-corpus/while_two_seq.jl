# while_two_seq(loss, u, a, v, w, i_n)
#
# Two sequential `while` loops. Each needs its own counter and its own
# count slot; one shared counter would give the second loop the first
# loop's trip count in the backward sweep.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: elementwise weight, length i_n
# v, w: scratch, length i_n
# i_n: length
function while_two_seq(loss, u, a, v, w, i_n)
    i_k = 1
    while i_k <= i_n
        v[i_k] = a[i_k] * u[i_k]
        i_k = i_k + 1
    end
    i_q = 1
    while i_q < i_n
        w[i_q] = v[i_q] * v[i_q + 1]
        i_q = i_q + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2 + w[i_x] ^ 2
    end
    return nothing
end
