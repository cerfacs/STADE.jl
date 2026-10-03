# while_for_inside(loss, u, a, v, i_n)
#
# A `for` nested in a `while`. The outer loop stays on the host because no
# launch can know its count, while the inner loop must still fuse and
# offload exactly as it would anywhere else.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: elementwise weight, length i_n
# v: scratch, length i_n
# i_n: length
function while_for_inside(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        for i_x = 1:i_n
            v[i_x] = v[i_x] + a[i_x] * u[i_x]
        end
        i_p = i_p + 1
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    return nothing
end
