# while_reduction(loss, u, a, v, i_n)
#
# A scalar reduction inside a `while` body. The reduction lowering added in
# v0.2.6 offers a matched loop both an atomic kernel and a mapreduce, and
# picks on the trip count. Here that inner loop sits inside a host loop
# whose own count is unknown, so the two mechanisms meet.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: elementwise weight, length i_n
# v: scratch, length i_n
# i_n: length
function while_reduction(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        s = 0.0
        for i_k = 1:i_n
            s = s + a[i_k] * u[i_k]
        end
        for i_x = 1:i_n
            v[i_x] = v[i_x] + s * u[i_x]
        end
        i_p = i_p + 1
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    return nothing
end
