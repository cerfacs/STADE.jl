function if_top_two_d(x, xd, y, yd, i_n)
    sd = 0.0
    s = 0.0
    if i_n > 2
        __cse_0 = x[2]
        __cse_1 = x[1]
        sd = __cse_0 * xd[1] + __cse_1 * xd[2]
        s = __cse_1 * __cse_0
    end
    td = 0.0
    t = 0.0
    if i_n > 3
        __cse_2 = x[3]
        __cse_3 = x[2]
        td = __cse_2 * xd[2] + __cse_3 * xd[3]
        t = __cse_3 * __cse_2
    end
    for i = 1:i_n
        __cse_4 = x[i]
        yd[i] = yd[i] + (((t * __cse_4) * sd + (s * __cse_4) * td) + (s * t) * xd[i])
        y[i] = y[i] + s * t * __cse_4
    end
    return nothing
end

function if_top_two(x, y, i_n)
    s = 0.0
    if i_n > 2
        s = x[1] * x[2]
    end
    t = 0.0
    if i_n > 3
        t = x[2] * x[3]
    end
    for i = 1:i_n
        y[i] = y[i] + s * t * x[i]
    end
end
