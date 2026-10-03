# mlp1d(loss, x, y, w1, b1, w2, b2, w3, b3, h1, h2, o, n_h)
#
# One-hidden-layer-per-stage MLP for a scalar regression, written for a
# single sample. 1 -> n_h -> n_h -> 1 with tanh activations and a squared
# error against the target.
#
# There is no batch loop here on purpose. bgen_ runs this kernel once per
# sample and accumulates the parameter gradients, so a batch loop would
# only grow the tape.
#
# loss: length-1 output array, accumulated in place
# x: input, length 1
# y: target, length 1
# w1, b1: first layer, length n_h each
# w2, b2: second layer, w2 is n_h * n_h row-major, b2 is n_h
# w3, b3: output layer, w3 is n_h, b3 is length 1
# h1, h2: hidden activations, length n_h, scratch
# o: prediction, length 1, scratch
# n_h: hidden width
function mlp1d(loss, x, y, w1, b1, w2, b2, w3, b3, h1, h2, o, n_h)
    for i_j = 1:n_h
        s1 = w1[i_j] * x[1] + b1[i_j]
        h1[i_j] = tanh(s1)
    end
    for i_j2 = 1:n_h
        s2 = b2[i_j2]
        for i_k = 1:n_h
            s2 = s2 + w2[(i_j2 - 1) * n_h + i_k] * h1[i_k]
        end
        h2[i_j2] = tanh(s2)
    end
    s3 = b3[1]
    for i_k2 = 1:n_h
        s3 = s3 + w3[i_k2] * h2[i_k2]
    end
    o[1] = s3
    loss[1] = loss[1] + (o[1] - y[1]) ^ 2
    return nothing
end
