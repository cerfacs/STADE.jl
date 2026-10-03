# while_zero(loss, u, a, v, i_n)
#
# The condition is false at entry whenever i_n is 0, so the loop runs zero
# times, the pushed count is 0, and the backward `for` must run zero times
# too. A loop that may not run has not run, in a new place: the count is a
# runtime value, so no bound proves it.
#
# The trailing loop is not decoration. Without it `v` is written and never
# read, so the whole `while` is dead code and the kernel would test
# nothing. It also gives `i_n` a use as a loop bound, without which shape
# inference has no reason to call it an integer: a `while` condition
# comparing two values says nothing about their kind.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: elementwise weight, length i_n
# v: scratch, length i_n
# i_n: length, and the argument that must be exercised at 0
function while_zero(loss, u, a, v, i_n)
    i_k = 1
    while i_k <= i_n
        v[i_k] = a[i_k] * u[i_k] + u[i_k]
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    return nothing
end
