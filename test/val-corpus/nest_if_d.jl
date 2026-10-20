function nest_if_d(x, xd, y, yd, i_n)
    for i = 1:i_n
        if i > 1
            if i > 2
                __cse_0 = x[i]
                __cse_1 = __cse_0 * xd[i]
                yd[i] = yd[i] + (__cse_1 + __cse_1)
                y[i] = y[i] + __cse_0 * __cse_0
            else
                __cse_2 = x[i]
                __cse_3 = (__cse_2 * __cse_2) * xd[i]
                yd[i] = yd[i] + ((__cse_3 + __cse_3) + __cse_3)
                y[i] = y[i] + __cse_2 * __cse_2 * __cse_2
            end
        end
    end
    return nothing
end

function nest_if(x, y, i_n)
    for i = 1:i_n
        if i > 1
            if i > 2
                y[i] = y[i] + x[i] * x[i]
            else
                y[i] = y[i] + x[i] * x[i] * x[i]
            end
        end
    end
end
