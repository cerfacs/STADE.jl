function retire_if_d(x, xd, y, yd, i_n, i_nlev, i_nrep, out, outd)
    curd = 0.0
    cur = i_n
    for i_l = 1:i_nlev
        for i_r = 1:i_nrep
            if i_r > 1
                for i = 1:cur
                    __cse_0 = x[i]
                    __cse_1 = __cse_0 * xd[i]
                    td = __cse_1 + __cse_1
                    t = __cse_0 * __cse_0
                    __cse_2 = t * td
                    yd[i] = yd[i] + (__cse_2 + __cse_2)
                    y[i] = y[i] + t * t
                end
            else
                for i = 1:cur
                    yd[i] = yd[i] + xd[i]
                    y[i] = y[i] + x[i]
                end
            end
        end
        curd = 0.0
        cur = div(cur + 1, 2)
    end
    outd[1] = yd[1]
    out[1] = y[1]
    return nothing
end

function retire_if(x, y, i_n, i_nlev, i_nrep, out)
    cur = i_n
    for i_l = 1:i_nlev
        for i_r = 1:i_nrep
            if i_r > 1
                for i = 1:cur
                    t = x[i] * x[i]
                    y[i] = y[i] + t * t
                end
            else
                for i = 1:cur
                    y[i] = y[i] + x[i]
                end
            end
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
end
