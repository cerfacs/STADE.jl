# so_inner(x, w, q, y, loss, i_n, i_nc)
#
# The tap loop of so_carry without the scalar accumulator: an outer loop carried by `run` (never split) around an inner loop that
# reassigns `xv` and `wv` twice per iteration and writes one array element per iteration (`q`). The inner loop is iteration-
# independent for the array check, and every read of `xv` and `wv` follows an assignment in the iteration, so only the push at the
# end of the body is missing: the first save of iteration `ci` holds the value iteration `ci - 1` ended with, which the reverse of
# iteration `ci - 1` reads. Splitting the inner loop would fill that slot with a placeholder. It must stay on the host.
#
# x: input, length i_nc + 1
# w: weights, length i_nc + 1
# q: scratch, length i_n * i_nc
# y: output, length i_n
# loss: length-1 output array, accumulated in place
# i_n, i_nc: trip counts
function so_inner(x, w, q, y, loss, i_n, i_nc)
    run = 0.0
    for t = 1:i_n
        for ci = 1:i_nc
            xv = x[ci]
            wv = w[ci]
            q[(t - 1) * i_nc + ci] = xv * wv
            xv = x[ci + 1]
            wv = w[ci + 1]
            q[(t - 1) * i_nc + ci] = q[(t - 1) * i_nc + ci] + xv * wv * run
        end
        run = run * 0.5 + q[(t - 1) * i_nc + 1]
        y[t] = run
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
    return nothing
end
