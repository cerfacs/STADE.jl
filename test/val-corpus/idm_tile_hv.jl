function initstacks_idm_tile_b(i_nt)
    return nothing
end

function idm_tile_hv(out, outb, inp, inpb, wt, wtb, bias, biasb, i_nt, i_cin, outd, outbd, inpd, inpbd, wtd, wtbd, biasd, biasbd)
    a0 = 0.0
    a1 = 0.0
    a0b = 0.0
    a1b = 0.0
    a0d = 0.0
    a0bd = 0.0
    a1d = 0.0
    a1bd = 0.0
    for i_t = 1:i_nt
        i_b = 2 * (i_t - 1)
        a0d = 0.0
        a0 = 0.0
        a1d = 0.0
        a1 = 0.0
        for i_c = 1:i_cin
            i_x = (i_c - 1) * (2i_nt + 1) + i_b
            __cse_0d = wtd[i_c]
            __cse_0 = wt[i_c]
            __cse_1d = inpd[i_x + 2]
            __cse_1 = inp[i_x + 2]
            __cse_2d = wtd[i_cin + i_c]
            __cse_2 = wt[i_cin + i_c]
            __hcse_0 = inp[i_x + 1]
            a0d = (a0d + (__cse_0 * inpd[i_x + 1] + __hcse_0 * __cse_0d)) + (__cse_2 * __cse_1d + __cse_1 * __cse_2d)
            a0 = a0 + __hcse_0 * __cse_0 + __cse_1 * __cse_2
            __hcse_1 = inp[i_x + 3]
            a1d = (a1d + (__cse_0 * __cse_1d + __cse_1 * __cse_0d)) + (__cse_2 * inpd[i_x + 3] + __hcse_1 * __cse_2d)
            a1 = a1 + __cse_1 * __cse_0 + __hcse_1 * __cse_2
        end
        __hcse_2 = a0 * a0d
        outd[i_b + 1] = (__hcse_2 + __hcse_2) + biasd[1]
        out[i_b + 1] = a0 * a0 + bias[1]
        __hcse_3 = a1 * a1d
        outd[i_b + 2] = (__hcse_3 + __hcse_3) + biasd[2]
        out[i_b + 2] = a1 * a1 + bias[2]
        i_b = 2 * (i_t - 1)
        __oldb_0d = outbd[i_b + 2]
        __oldb_0 = outb[i_b + 2]
        outbd[i_b + 2] = 0.0
        outb[i_b + 2] = 0.0
        __cse_3d = __oldb_0 * a1d + a1 * __oldb_0d
        __cse_3 = a1 * __oldb_0
        a1bd = a1bd + __cse_3d
        a1b = a1b + __cse_3
        a1bd = a1bd + __cse_3d
        a1b = a1b + __cse_3
        biasbd[2] = biasbd[2] + __oldb_0d
        biasb[2] = biasb[2] + __oldb_0
        __oldb_0d = outbd[i_b + 1]
        __oldb_0 = outb[i_b + 1]
        outbd[i_b + 1] = 0.0
        outb[i_b + 1] = 0.0
        __cse_4d = __oldb_0 * a0d + a0 * __oldb_0d
        __cse_4 = a0 * __oldb_0
        a0bd = a0bd + __cse_4d
        a0b = a0b + __cse_4
        a0bd = a0bd + __cse_4d
        a0b = a0b + __cse_4
        biasbd[1] = biasbd[1] + __oldb_0d
        biasb[1] = biasb[1] + __oldb_0
        for i_c = 1:i_cin
            __icse_5 = (i_c - 1) * (2i_nt + 1) + i_b
            i_x = __icse_5
            __cse_6d = wtd[i_c]
            __cse_6 = wt[i_c]
            __cse_7d = inpd[i_x + 2]
            __cse_7 = inp[i_x + 2]
            __cse_8d = wtd[i_cin + i_c]
            __cse_8 = wt[i_cin + i_c]
            __hcse_4 = inp[i_x + 1]
            a0d = (a0d + (__cse_6 * inpd[i_x + 1] + __hcse_4 * __cse_6d)) + (__cse_8 * __cse_7d + __cse_7 * __cse_8d)
            a0 = a0 + __hcse_4 * __cse_6 + __cse_7 * __cse_8
            __hcse_5 = inp[i_x + 3]
            a1d = (a1d + (__cse_6 * __cse_7d + __cse_7 * __cse_6d)) + (__cse_8 * inpd[i_x + 3] + __hcse_5 * __cse_8d)
            a1 = a1 + __cse_7 * __cse_6 + __hcse_5 * __cse_8
            i_x = __icse_5
            inpbd[i_x + 2] = inpbd[i_x + 2] + (a1b * __cse_6d + __cse_6 * a1bd)
            inpb[i_x + 2] = inpb[i_x + 2] + __cse_6 * a1b
            __cse_9d = inpd[i_x + 2]
            __cse_9 = inp[i_x + 2]
            wtbd[i_c] = wtbd[i_c] + (a1b * __cse_9d + __cse_9 * a1bd)
            wtb[i_c] = wtb[i_c] + __cse_9 * a1b
            inpbd[i_x + 3] = inpbd[i_x + 3] + (a1b * __cse_8d + __cse_8 * a1bd)
            inpb[i_x + 3] = inpb[i_x + 3] + __cse_8 * a1b
            __hcse_6 = inp[i_x + 3]
            wtbd[i_cin + i_c] = wtbd[i_cin + i_c] + (a1b * inpd[i_x + 3] + __hcse_6 * a1bd)
            wtb[i_cin + i_c] = wtb[i_cin + i_c] + __hcse_6 * a1b
            inpbd[i_x + 1] = inpbd[i_x + 1] + (a0b * __cse_6d + __cse_6 * a0bd)
            inpb[i_x + 1] = inpb[i_x + 1] + __cse_6 * a0b
            __hcse_7 = inp[i_x + 1]
            wtbd[i_c] = wtbd[i_c] + (a0b * inpd[i_x + 1] + __hcse_7 * a0bd)
            wtb[i_c] = wtb[i_c] + __hcse_7 * a0b
            inpbd[i_x + 2] = inpbd[i_x + 2] + (a0b * __cse_8d + __cse_8 * a0bd)
            inpb[i_x + 2] = inpb[i_x + 2] + __cse_8 * a0b
            wtbd[i_cin + i_c] = wtbd[i_cin + i_c] + (a0b * __cse_9d + __cse_9 * a0bd)
            wtb[i_cin + i_c] = wtb[i_cin + i_c] + __cse_9 * a0b
        end
        a1bd = 0.0
        a1b = 0.0
        a0bd = 0.0
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
