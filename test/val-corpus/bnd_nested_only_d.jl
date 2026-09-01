function bnd_nested_only_d(x, xd, y, yd, i_n, i_m, out, outd, acc, accd)
    vd = 0.0
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            __cse_0 = y[i_i]
            __cse_1 = x[i_j]
            vd = __cse_0 * xd[i_j] + __cse_1 * yd[i_i]
            v = __cse_1 * __cse_0
            __cse_2 = v * vd
            outd[i_i] = outd[i_i] + (__cse_2 + __cse_2)
            out[i_i] = out[i_i] + v * v
        end
        __cse_3 = y[i_i]
        __cse_4 = out[i_i - 1]
        outd[i_i] = outd[i_i] + (__cse_3 * outd[i_i - 1] + __cse_4 * yd[i_i])
        out[i_i] = out[i_i] + __cse_4 * __cse_3
    end
    __cse_5 = y[1]
    __cse_6 = x[1]
    vd = __cse_5 * xd[1] + __cse_6 * yd[1]
    v = __cse_6 * __cse_5
    __cse_7 = v * vd
    accd[1] = accd[1] + (__cse_7 + __cse_7)
    acc[1] = acc[1] + v * v
    return nothing
end

function bnd_nested_only(x, y, i_n, i_m, out, acc)
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            v = x[i_j] * y[i_i]
            out[i_i] = out[i_i] + v * v
        end
        out[i_i] = out[i_i] + out[i_i - 1] * y[i_i]
    end
    v = x[1] * y[1]
    acc[1] = acc[1] + v * v
end
