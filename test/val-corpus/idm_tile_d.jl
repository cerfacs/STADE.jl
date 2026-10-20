function idm_tile_d(out, outd, inp, inpd, wt, wtd, bias, biasd, i_nt, i_cin)
    for i_t = 1:i_nt
        i_bd = 0.0
        i_b = 2 * (i_t - 1)
        a0d = 0.0
        a0 = 0.0
        a1d = 0.0
        a1 = 0.0
        for i_c = 1:i_cin
            i_xd = 0.0
            i_x = (i_c - 1) * (2i_nt + 1) + i_b
            __cse_0 = wt[i_c]
            __cse_1 = inp[i_x + 1]
            __cse_2 = wtd[i_c]
            __cse_3 = wt[i_cin + i_c]
            __cse_4 = inpd[i_x + 2]
            __cse_5 = inp[i_x + 2]
            __cse_6 = wtd[i_cin + i_c]
            a0d = (a0d + (__cse_0 * inpd[i_x + 1] + __cse_1 * __cse_2)) + (__cse_3 * __cse_4 + __cse_5 * __cse_6)
            a0 = a0 + __cse_1 * __cse_0 + __cse_5 * __cse_3
            __cse_7 = inp[i_x + 3]
            a1d = (a1d + (__cse_0 * __cse_4 + __cse_5 * __cse_2)) + (__cse_3 * inpd[i_x + 3] + __cse_7 * __cse_6)
            a1 = a1 + __cse_5 * __cse_0 + __cse_7 * __cse_3
        end
        __cse_8 = a0 * a0d
        outd[i_b + 1] = (__cse_8 + __cse_8) + biasd[1]
        out[i_b + 1] = a0 * a0 + bias[1]
        __cse_9 = a1 * a1d
        outd[i_b + 2] = (__cse_9 + __cse_9) + biasd[2]
        out[i_b + 2] = a1 * a1 + bias[2]
    end
    return nothing
end

function idm_tile(out, inp, wt, bias, i_nt, i_cin)
    for i_t = 1:i_nt
        i_b = 2 * (i_t - 1)
        a0 = 0.0
        a1 = 0.0
        for i_c = 1:i_cin
            i_x = (i_c - 1) * (2i_nt + 1) + i_b
            a0 = a0 + inp[i_x + 1] * wt[i_c] + inp[i_x + 2] * wt[i_cin + i_c]
            a1 = a1 + inp[i_x + 2] * wt[i_c] + inp[i_x + 3] * wt[i_cin + i_c]
        end
        out[i_b + 1] = a0 * a0 + bias[1]
        out[i_b + 2] = a1 * a1 + bias[2]
    end
end
