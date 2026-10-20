# if_scalar_rewrite(x, w, y, loss, i_n)
#
# `wv` is assigned in an `if`, read, then unconditionally rewritten and read again. The backward sweep used to recompute `wv` from the
# branch flag at the top of the block (for reads after the `if`), which clobbered the final `wv` the second read needs: the adjoint was
# wrong (relative error 1e-1 and up) under every flag combination while the tangent and its finite differences were right. A later
# write to the same scalar in the block restores the post-`if` value at its own reverse position, so no up-front recompute is wanted.
#
# x: input, length i_n
# w: weights, length i_n + 1
# y: output, length i_n
# loss: length-1 output array, accumulated in place
# i_n: tile count
function if_scalar_rewrite(x, w, y, loss, i_n)
    for t = 1:i_n
        wv = 0.0
        if x[t] > 0.0
            wv = w[t]
        end
        y[t] = wv * wv * x[t]
        wv = w[t + 1]
        y[t] = y[t] + wv * wv * x[t]
    end
    for m = 1:i_n
        loss[1] = loss[1] + y[m] * y[m]
    end
    return nothing
end
