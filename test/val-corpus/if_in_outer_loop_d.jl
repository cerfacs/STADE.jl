function if_in_outer_loop_d(x, xd, y, yd, i_n)
    for i_k = 1:2
        sd = 0.0
        s = 0.0
        if i_n > 2
            __cse_0 = x[i_k + 1]
            __cse_1 = x[i_k]
            sd = __cse_0 * xd[i_k] + __cse_1 * xd[i_k + 1]
            s = __cse_1 * __cse_0
        end
        for i = 1:i_n
            __cse_2 = x[i]
            __cse_3 = (s * __cse_2) * sd
            yd[i] = yd[i] + ((__cse_3 + __cse_3) + (s * s) * xd[i])
            y[i] = y[i] + s * s * __cse_2
        end
    end
    return nothing
end

function if_in_outer_loop(x, y, i_n)
    for i_k = 1:2
        s = 0.0
        if i_n > 2
            s = x[i_k] * x[i_k + 1]
        end
        for i = 1:i_n
            y[i] = y[i] + s * s * x[i]
        end
    end
end
