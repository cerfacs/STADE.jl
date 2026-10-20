function if_scalar_then_loop_if_d(u, ud, x, xd, y, yd, rhs, rhsd, i_n)
    for i = 1:i_n
        leftd = 0.0
        left = 0.0
        if i > 1
            leftd = ud[i - 1]
            left = u[i - 1]
        end
        for j = 1:2
            if j > 1
                __cse_0 = x[j]
                __cse_1 = __cse_0 * xd[j]
                yd[i] = yd[i] + (__cse_1 + __cse_1)
                y[i] = y[i] + __cse_0 * __cse_0
            end
        end
        __cse_2 = rhs[i]
        __cse_3 = (__cse_2 * left) * leftd
        ud[i] = ud[i] + (((left * left) * rhsd[i] + __cse_3) + __cse_3)
        u[i] = u[i] + __cse_2 * left * left
    end
    return nothing
end

function if_scalar_then_loop_if(u, x, y, rhs, i_n)
    for i = 1:i_n
        left = 0.0
        if i > 1
            left = u[i - 1]
        end
        for j = 1:2
            if j > 1
                y[i] = y[i] + x[j] * x[j]
            end
        end
        u[i] = u[i] + rhs[i] * left * left
    end
end
