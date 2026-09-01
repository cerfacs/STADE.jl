function ii_readbefore_d(x, xd, i_n, i_m, out, outd)
    __cse_0 = x[1]
    __cse_1 = __cse_0 * xd[1]
    sd = __cse_1 + __cse_1
    s = __cse_0 * __cse_0
    __cse_2 = s * sd
    outd[1] = outd[1] + (__cse_2 + __cse_2)
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        sd = 0.0
        s = 0.0
        for i_j = 1:i_m
            __cse_3 = x[i_i]
            __cse_4 = x[i_j]
            sd = sd + (__cse_3 * xd[i_j] + __cse_4 * xd[i_i])
            s = s + __cse_4 * __cse_3
        end
        __cse_5 = s * sd
        outd[i_i] = outd[i_i] + (__cse_5 + __cse_5)
        out[i_i] = out[i_i] + s * s
    end
    return nothing
end

function ii_readbefore(x, i_n, i_m, out)
    s = x[1] * x[1]
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_j] * x[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
end
