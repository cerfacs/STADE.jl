function initstacks_geomrecur_b(i_n)
    u_stack = Vector{Float64}(undef, max(0, div(i_n - 2, 1) + 1))
    return u_stack
end

function geomrecur_b(loss, lossb, u, ub, c, cb, i_n, u_stack)
    for i_x = 2:i_n
        __idx_u_stack_0 = (i_x - 2) + 1
        u_stack[__idx_u_stack_0] = u[i_x]
        u[i_x] = c * u[i_x - 1]
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * lossb[1]
    end
    for i_x = i_n:-1:2
        __idx_u_stack_0 = (i_x - 2) + 1
        u[i_x] = u_stack[__idx_u_stack_0]
        cb = cb + u[i_x - 1] * ub[i_x]
        ub[i_x - 1] = ub[i_x - 1] + c * ub[i_x]
        ub[i_x] = 0.0
    end
    return cb
end

function geomrecur(loss, u, c, i_n)
    for i_x = 2:i_n
        u[i_x] = c * u[i_x - 1]
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
end
