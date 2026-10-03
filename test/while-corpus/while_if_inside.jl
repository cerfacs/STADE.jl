# while_if_inside(loss, u, a, v, i_n)
#
# An `:if` inside a `while`. Two snapshot kinds interleave at one site: a
# branch flag per iteration and the loop's own count. Neither may overwrite
# the other's slot.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: elementwise weight, length i_n
# v: scratch, length i_n
# i_n: length
function while_if_inside(loss, u, a, v, i_n)
    i_k = 1
    while i_k <= i_n
        if u[i_k] > 0.0
            v[i_k] = a[i_k] * u[i_k]
        else
            v[i_k] = a[i_k] * u[i_k] * u[i_k]
        end
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    return nothing
end
