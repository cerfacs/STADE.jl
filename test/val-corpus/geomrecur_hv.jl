function initstacks_geomrecur_b(i_n)
    u_stack = Vector{Float64}(undef, max(0, div(i_n - 2, 1) + 1))
    return u_stack
end

function geomrecur_hv(loss, lossb, u, ub, c, cb, i_n, lossd, lossbd, ud, ubd, cd, cbd, u_stack)
    u_stack_d = Vector{Float64}(undef, length(u_stack))
    for i_x = 2:i_n
        __idx_u_stack_0 = (i_x - 2) + 1
        u_stack_d[__idx_u_stack_0] = ud[i_x]
        u_stack[__idx_u_stack_0] = u[i_x]
        __cse_1 = u[i_x - 1]
        ud[i_x] = __cse_1 * cd + c * ud[i_x - 1]
        u[i_x] = c * __cse_1
    end
    for i_x = 1:i_n
        __cse_2 = u[i_x]
        lossd[1] = lossd[1] + (2__cse_2) * ud[i_x]
        loss[1] = loss[1] + __cse_2 ^ 2
    end
    for i_x = i_n:-1:1
        __cse_3 = lossb[1]
        __cse_4 = 2 * u[i_x]
        ubd[i_x] = ubd[i_x] + (__cse_3 * (2 * ud[i_x]) + __cse_4 * lossbd[1])
        ub[i_x] = ub[i_x] + __cse_4 * __cse_3
    end
    for i_x = i_n:-1:2
        __idx_u_stack_0 = (i_x - 2) + 1
        ud[i_x] = u_stack_d[__idx_u_stack_0]
        u[i_x] = u_stack[__idx_u_stack_0]
        __cse_0d = ubd[i_x]
        __cse_0 = ub[i_x]
        __cse_5 = u[i_x - 1]
        cbd = cbd + (__cse_0 * ud[i_x - 1] + __cse_5 * __cse_0d)
        cb = cb + __cse_5 * __cse_0
        ubd[i_x - 1] = ubd[i_x - 1] + (__cse_0 * cd + c * __cse_0d)
        ub[i_x - 1] = ub[i_x - 1] + c * __cse_0
        ubd[i_x] = 0.0
        ub[i_x] = 0.0
    end
    return (cb, cbd)
end

function geomrecur(loss, u, c, i_n)
    for i_x = 2:i_n
        u[i_x] = c * u[i_x - 1]
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
end
