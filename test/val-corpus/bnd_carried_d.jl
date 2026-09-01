function bnd_carried_d(x, xd, y, yd, i_n, out, outd)
    td = 0.0
    t = 1.0
    for i_i = 1:i_n
        __cse_0 = y[i_i]
        __cse_1 = x[i_i]
        td = __cse_0 * xd[i_i] + __cse_1 * yd[i_i]
        t = __cse_1 * __cse_0
        __cse_2 = t * td
        outd[i_i] = outd[i_i] + (__cse_2 + __cse_2)
        out[i_i] = out[i_i] + t * t
    end
    return nothing
end

function bnd_carried(x, y, i_n, out)
    t = 1.0
    for i_i = 1:i_n
        t = x[i_i] * y[i_i]
        out[i_i] = out[i_i] + t * t
    end
end
