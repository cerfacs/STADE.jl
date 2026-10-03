# while_nested_while(loss, u, a, v, i_n)
#
# A `while` inside a `while`. The inner count is ragged against the outer
# one, a level deeper than while_in_for, where the outer count was itself a
# closed-form `for` bound. Nothing in the count layout may assume an
# ancestor has a known trip count.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: elementwise weight, length i_n
# v: scratch, length i_n
# i_n: length
function while_nested_while(loss, u, a, v, i_n)
    i_p = 1
    while i_p <= 2
        i_k = 1
        while i_k <= i_n
            v[i_k] = v[i_k] + a[i_k] * u[i_k] * i_p
            i_k = i_k + 1
        end
        i_p = i_p + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    return nothing
end
