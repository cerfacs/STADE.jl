function initstacks_advection_b(i_nnode, i_nstep)
    du_stack = Vector{Float64}(undef, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_b(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    for i_ = 1:i_nstep
        for i_x = 2:i_nnode
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
            du_stack[__idx_du_stack_0] = du[i_x]
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x = i_nnode:-1:2
            __cse_0 = du[i_x]
            __cse_1 = -(ub[i_x])
            __cse_2 = (1.0 / dx) * __cse_1
            cb = cb + (dt * __cse_0) * __cse_2
            dtb = dtb + (c * __cse_0) * __cse_2
            dub[i_x] = dub[i_x] + (c * dt) * __cse_2
            dxb = dxb + -((c * dt * __cse_0) / dx ^ 2) * __cse_1
        end
        for i_x = i_nnode:-1:2
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
            du[i_x] = du_stack[__idx_du_stack_0]
            __oldb_2 = dub[i_x]
            dub[i_x] = 0.0
            ub[i_x] = ub[i_x] + __oldb_2
            ub[i_x - 1] = ub[i_x - 1] + -__oldb_2
        end
    end
    return (cb, dxb, dtb)
end

function advection(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_ = 1:i_nstep
        for i_x = 2:i_nnode
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
end
