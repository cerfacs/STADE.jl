# idm_slots(loss, y, x, w, i_n)
#
# Offload regression for the reverse sweep of a loop whose iteration writes
# several slots of one array. Iteration i_t owns four consecutive slots of y
# and reads two neighbouring elements of x. The loop is independent, so it
# runs as one GPU thread per i_t.
#
# The adjoint of this loop is the shape that a register-tiled kernel
# produces. y's shadow is consumed at four slot expressions that differ by a
# constant offset, and x's shadow is accumulated at two expressions that
# overlap between neighbouring iterations. Neither idiom is a recurrence.
#
# loss: length-1 output array, accumulated in place
# y: output array of length 4 * i_n
# x: input array of length i_n + 1
# w: weight array of length 3
# i_n: number of iterations
function idm_slots(loss, y, x, w, i_n)
    for i_t = 1:i_n
        i_b = 4 * (i_t - 1)
        # four output slots, each reads both neighbours
        y[i_b + 1] = x[i_t] * w[1] + x[i_t + 1] * w[2]
        y[i_b + 2] = x[i_t] * w[2] + x[i_t + 1] * w[3]
        y[i_b + 3] = x[i_t] * w[3] + x[i_t + 1] * w[1]
        y[i_b + 4] = x[i_t] * w[1] + x[i_t + 1] * w[3]
    end
    # plain quadratic loss over the output
    for i_k = 1:4 * i_n
        loss[1] = loss[1] + y[i_k] * y[i_k]
    end
    return nothing
end
