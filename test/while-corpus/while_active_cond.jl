# while_active_cond(loss, u, a, v, i_n)
#
# The condition reads a float the body updates, so the trip count depends
# on the input values. That makes the count a step function of the inputs:
# constant almost everywhere, jumping where the loop takes one more turn.
#
# STADE replays the recorded count, so the gradient is the derivative at
# FIXED trip count. That is correct almost everywhere and wrong at a jump.
# Every AD tool behaves this way. This kernel exists so the behaviour is
# exercised rather than assumed, and the finite-difference oracle must be
# read with that in mind.
#
# loss: length-1 output array, accumulated in place
# u: input, length i_n
# a: decay factor, length i_n
# v: scratch, length i_n
# i_n: length
function while_active_cond(loss, u, a, v, i_n)
    s = 1.0
    i_k = 1
    while s > 0.05 && i_k <= i_n
        s = s * (0.5 + 0.25 * a[i_k] * a[i_k])
        v[i_k] = s * u[i_k]
        i_k = i_k + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2
    end
    return nothing
end
