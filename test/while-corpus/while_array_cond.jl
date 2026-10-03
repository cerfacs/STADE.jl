# while_array_cond(loss, u, a, v, i_n)
#
# The condition reads an array element the body writes. The backward sweep
# never re-evaluates the condition, so this checks that the values the
# condition consumed are snapshotted rather than rebuilt: a `while` has no
# loop variable, so nothing can be recomputed from an index.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: elementwise weight, length i_n
# v: scratch, caller-visible, length i_n
# i_n: length
function while_array_cond(loss, u, a, v, i_n)
    i_k = 1
    v[1] = 1.0
    while v[i_k] > 0.1 && i_k < i_n
        v[i_k + 1] = 0.5 * v[i_k] * (1.0 + a[i_k] * a[i_k])
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + (v[i_x] * u[i_x]) ^ 2
    end
    return nothing
end
