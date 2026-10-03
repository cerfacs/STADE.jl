# while_in_for(loss, u, a, v, i_n, i_m)
#
# A `while` nested in a `for`. Each outer iteration produces its own count,
# so the count stack is ragged and cannot be one scalar. This is the Tier B
# shape, and the reason a `:whilecount` site is per-execution rather than
# per-statement.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: elementwise weight, length i_n
# v: scratch, length i_n
# i_n: inner length
# i_m: outer count
function while_in_for(loss, u, a, v, i_n, i_m)
    for i_j = 1:i_m
        i_k = 1
        while i_k <= i_n
            v[i_k] = a[i_k] * u[i_k] + v[i_k]
            i_k = i_k + 1
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    return nothing
end
