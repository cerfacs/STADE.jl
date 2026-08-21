function initstacks_geomrecur_b(i_n)
    u_stack = Vector{Float64}(undef, div(i_n - 2, 1) + 1)
    return u_stack
end

function geomrecur_hv(loss, lossb, u, ub, c, cb, i_n, lossd, lossbd, ud, ubd, cd, cbd, u_stack)
    u_stack_d = Vector{Float64}(undef, length(u_stack))
    for i_seq_x = 2:i_n
        u_stack_d[(i_seq_x - 2) + 1] = ud[i_seq_x]
        u_stack[(i_seq_x - 2) + 1] = u[i_seq_x]
        ud[i_seq_x] = u[i_seq_x - 1] * cd + c * ud[i_seq_x - 1]
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * u[i_seq_x]) * ud[i_seq_x]
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        ubd[i_seq_x] = ubd[i_seq_x] + (lossb[1] * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * lossbd[1])
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * lossb[1]
    end
    for i_seq_x = i_n:-1:2
        ud[i_seq_x] = u_stack_d[(i_seq_x - 2) + 1]
        u[i_seq_x] = u_stack[(i_seq_x - 2) + 1]
        cbd = cbd + (ub[i_seq_x] * ud[i_seq_x - 1] + u[i_seq_x - 1] * ubd[i_seq_x])
        cb = cb + u[i_seq_x - 1] * ub[i_seq_x]
        ubd[i_seq_x - 1] = ubd[i_seq_x - 1] + (ub[i_seq_x] * cd + c * ubd[i_seq_x])
        ub[i_seq_x - 1] = ub[i_seq_x - 1] + c * ub[i_seq_x]
        ubd[i_seq_x] = 0.0
        ub[i_seq_x] = 0.0
    end
    return (cb, cbd)
end

function geomrecur(loss, u, c, i_n)
    for i_seq_x = 2:i_n
        u[i_seq_x] = c * u[i_seq_x - 1]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + u[i_seq_x] ^ 2
    end
end
