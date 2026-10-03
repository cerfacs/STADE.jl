# ddp_twoparam(loss, u, w, b, h, o, i_n, i_m)
#
# Two parameter groups of different shapes: a dense weight w of length
# i_m * i_n and a bias b of length i_m. bgen_ must emit one collective
# per group, so the delete-one-collective measurement has more than one
# call to remove. A single-parameter kernel cannot distinguish "the
# collectives work" from "the one collective works".
#
# loss: length-1 output array, accumulated in place
# u: input features for one sample, length i_n
# w: dense weight, row-major, length i_m * i_n
# b: bias, length i_m
# h: scratch activations, length i_m
# o: scratch residuals, length i_m
# i_n: input feature count
# i_m: output feature count
function ddp_twoparam(loss, u, w, b, h, o, i_n, i_m)
    for i_j = 1:i_m
        s = b[i_j]
        for i_k = 1:i_n
            s = s + w[(i_j - 1) * i_n + i_k] * u[i_k]
        end
        h[i_j] = s
    end
    for i_j2 = 1:i_m
        o[i_j2] = h[i_j2] - u[1]
    end
    for i_j3 = 1:i_m
        loss[1] = loss[1] + o[i_j3] ^ 2
    end
    return nothing
end
