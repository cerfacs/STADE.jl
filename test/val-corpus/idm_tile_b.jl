function initstacks_idm_tile_b(i_nt)
    return nothing
end

function idm_tile_b(out, outb, inp, inpb, wt, wtb, bias, biasb, i_nt, i_cin)
    a0 = 0.0
    a1 = 0.0
    a0b = 0.0
    a1b = 0.0
    for i_t = 1:i_nt
        i_b = 2 * (i_t - 1)
        a0 = 0.0
        a1 = 0.0
        for i_c = 1:i_cin
            i_x = (i_c - 1) * (2i_nt + 1) + i_b
            __cse_0 = wt[i_c]
            __cse_1 = inp[i_x + 2]
            __cse_2 = wt[i_cin + i_c]
            a0 = a0 + inp[i_x + 1] * __cse_0 + __cse_1 * __cse_2
            a1 = a1 + __cse_1 * __cse_0 + inp[i_x + 3] * __cse_2
        end
        out[i_b + 1] = a0 * a0 + bias[1]
        out[i_b + 2] = a1 * a1 + bias[2]
        i_b = 2 * (i_t - 1)
        __oldb_0 = outb[i_b + 2]
        outb[i_b + 2] = 0.0
        __cse_3 = a1 * __oldb_0
        a1b = a1b + __cse_3
        a1b = a1b + __cse_3
        biasb[2] = biasb[2] + __oldb_0
        __oldb_0 = outb[i_b + 1]
        outb[i_b + 1] = 0.0
        __cse_4 = a0 * __oldb_0
        a0b = a0b + __cse_4
        a0b = a0b + __cse_4
        biasb[1] = biasb[1] + __oldb_0
        for i_c = 1:i_cin
            __icse_5 = (i_c - 1) * (2i_nt + 1) + i_b
            i_x = __icse_5
            __cse_6 = wt[i_c]
            __cse_7 = inp[i_x + 2]
            __cse_8 = wt[i_cin + i_c]
            a0 = a0 + inp[i_x + 1] * __cse_6 + __cse_7 * __cse_8
            a1 = a1 + __cse_7 * __cse_6 + inp[i_x + 3] * __cse_8
            i_x = __icse_5
            inpb[i_x + 2] = inpb[i_x + 2] + __cse_6 * a1b
            __cse_9 = inp[i_x + 2]
            wtb[i_c] = wtb[i_c] + __cse_9 * a1b
            inpb[i_x + 3] = inpb[i_x + 3] + __cse_8 * a1b
            wtb[i_cin + i_c] = wtb[i_cin + i_c] + inp[i_x + 3] * a1b
            inpb[i_x + 1] = inpb[i_x + 1] + __cse_6 * a0b
            wtb[i_c] = wtb[i_c] + inp[i_x + 1] * a0b
            inpb[i_x + 2] = inpb[i_x + 2] + __cse_8 * a0b
            wtb[i_cin + i_c] = wtb[i_cin + i_c] + __cse_9 * a0b
        end
        a1b = 0.0
        a0b = 0.0
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
