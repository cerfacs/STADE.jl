function coarsen_retire_d(x, xd, y, yd, n, levels, out, outd)
    curd = 0.0
    cur = n
    for i_l = 1:levels
        for i = 1:cur
            __cse_0 = x[i]
            __cse_1 = __cse_0 * xd[i]
            td = __cse_1 + __cse_1
            t = __cse_0 * __cse_0
            __cse_2 = t * td
            yd[i] = yd[i] + (__cse_2 + __cse_2)
            y[i] = y[i] + t * t
        end
        curd = 0.0
        cur = div(cur + 1, 2)
    end
    outd[1] = yd[1]
    out[1] = y[1]
    return nothing
end

function coarsen_retire(x, y, n, levels, out)
    cur = n
    for i_l = 1:levels
        for i = 1:cur
            t = x[i] * x[i]
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
end
