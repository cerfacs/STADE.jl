function red_escape_d(x, xd, y, yd, i_n, i_m, out, outd, acc, accd)
    sd = 0.0
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            __cse_0 = y[i_i]
            __cse_1 = x[i_j]
            sd = sd + (__cse_0 * xd[i_j] + __cse_1 * yd[i_i])
            s = s + __cse_1 * __cse_0
        end
        __cse_2 = s * sd
        outd[i_i] = outd[i_i] + (__cse_2 + __cse_2)
        out[i_i] = out[i_i] + s * s
    end
    __cse_3 = y[1]
    __cse_4 = x[1]
    sd = __cse_3 * xd[1] + __cse_4 * yd[1]
    s = __cse_4 * __cse_3
    __cse_5 = s * sd
    accd[1] = accd[1] + (__cse_5 + __cse_5)
    acc[1] = acc[1] + s * s
    return nothing
end

function red_escape(x, y, i_n, i_m, out, acc)
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            s = s + x[i_j] * y[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
    s = x[1] * y[1]
    acc[1] = acc[1] + s * s
end
