function if_top_scalar_d(x, xd, y, yd, i_n)
    if x[1] > 0.0
        __cse_0 = x[2]
        __cse_1 = x[1]
        sd = __cse_0 * xd[1] + __cse_1 * xd[2]
        s = __cse_1 * __cse_0
    else
        __cse_2 = x[2]
        __cse_3 = __cse_2 * xd[2]
        sd = __cse_3 + __cse_3
        s = __cse_2 * __cse_2
    end
    for i = 1:i_n
        __cse_4 = x[i]
        __cse_5 = (s * __cse_4) * sd
        yd[i] = yd[i] + ((__cse_5 + __cse_5) + (s * s) * xd[i])
        y[i] = y[i] + s * s * __cse_4
    end
    return nothing
end

function if_top_scalar(x, y, i_n)
    if x[1] > 0.0
        s = x[1] * x[2]
    else
        s = x[2] * x[2]
    end
    for i = 1:i_n
        y[i] = y[i] + s * s * x[i]
    end
end
