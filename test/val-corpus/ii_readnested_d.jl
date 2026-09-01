function ii_readnested_d(x, xd, y, yd, i_n, i_m, out, outd, acc, accd)
    vd = 0.0
    v = 1.0
    for i_i = 1:i_n
        __cse_0 = x[i_i]
        __cse_1 = __cse_0 * xd[i_i]
        vd = __cse_1 + __cse_1
        v = __cse_0 * __cse_0
        __cse_2 = v * vd
        accd[i_i] = accd[i_i] + (__cse_2 + __cse_2)
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            __cse_3 = y[i_j]
            __cse_4 = x[i_i]
            vd = __cse_3 * xd[i_i] + __cse_4 * yd[i_j]
            v = __cse_4 * __cse_3
            __cse_5 = v * vd
            outd[i_i] = outd[i_i] + (__cse_5 + __cse_5)
            out[i_i] = out[i_i] + v * v
        end
    end
    return nothing
end

function ii_readnested(x, y, i_n, i_m, out, acc)
    v = 1.0
    for i_i = 1:i_n
        v = x[i_i] * x[i_i]
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            v = x[i_i] * y[i_j]
            out[i_i] = out[i_i] + v * v
        end
    end
end
