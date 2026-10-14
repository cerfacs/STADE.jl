# idm_tile(out, inp, wt, bias, i_nt, i_cin)
#
# Miniature register-tiled convolution. Iteration i_t owns two adjacent
# output slots and keeps one scalar accumulator per slot over an inner
# loop on the input channel. The adjoint carries the full tiled shape: the
# output shadow is consumed at two slots, the input shadow is accumulated
# at two slots per channel, the weight shadow is accumulated at an index
# that every iteration shares, and the accumulator shadows reset at the
# end of each iteration.
#
# out: output array of length 2 * i_nt
# inp: input array of length i_cin * (2 * i_nt + 1)
# wt: weight array of length 2 * i_cin
# bias: bias array of length 2
# i_nt: number of tiles
# i_cin: number of input channels
function idm_tile(out, inp, wt, bias, i_nt, i_cin)
    for i_t = 1:i_nt
        i_b = 2 * (i_t - 1)
        a0 = 0.0
        a1 = 0.0
        for i_c = 1:i_cin
            i_x = (i_c - 1) * (2 * i_nt + 1) + i_b
            a0 = a0 + inp[i_x + 1] * wt[i_c] + inp[i_x + 2] * wt[i_cin + i_c]
            a1 = a1 + inp[i_x + 2] * wt[i_c] + inp[i_x + 3] * wt[i_cin + i_c]
        end
        out[i_b + 1] = a0 * a0 + bias[1]
        out[i_b + 2] = a1 * a1 + bias[2]
    end
    return nothing
end
