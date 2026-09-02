function cse_intoffset_d(out, outd, u, ud, i_nrow, i_base)
    for i_r = 1:i_nrow
        i_lend = 0.0
        i_len = i_base + i_r
        i_offd = 0.0
        __icse_0 = (i_r - 1) * i_base
        __icse_1 = __icse_0 + i_len
        i_off = __icse_1
        i_lod = 0.0
        i_lo = __icse_1 - i_len
        i_hid = 0.0
        i_hi = __icse_0 + i_len + 1
        sd = 0.0
        s = 0.0
        for i_j = 1:i_len
            __cse_2 = u[i_lo + i_j]
            __cse_3 = u[(i_r - 1) * i_base + i_j]
            sd = sd + (__cse_2 * ud[(i_r - 1) * i_base + i_j] + __cse_3 * ud[i_lo + i_j])
            s = s + __cse_3 * __cse_2
        end
        __cse_4 = u[i_off]
        __cse_5 = u[i_hi]
        outd[i_r] = (__cse_4 * sd + s * ud[i_off]) + (__cse_5 * sd + s * ud[i_hi])
        out[i_r] = s * __cse_4 + s * __cse_5
    end
    return nothing
end

function cse_intoffset(out, u, i_nrow, i_base)
    for i_r = 1:i_nrow
        i_len = i_base + i_r
        i_off = (i_r - 1) * i_base + i_len
        i_lo = ((i_r - 1) * i_base + i_len) - i_len
        i_hi = (i_r - 1) * i_base + i_len + 1
        s = 0.0
        for i_j = 1:i_len
            s = s + u[(i_r - 1) * i_base + i_j] * u[i_lo + i_j]
        end
        out[i_r] = s * u[i_off] + s * u[i_hi]
    end
end
