function initstacks_advection_b(i_nnode, i_nstep)
    du_stack = Vector{Float64}(undef, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_hv(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    du_stack_d = Vector{Float64}(undef, length(du_stack))
    for i_ = 1:i_nstep
        for i_x = 2:i_nnode
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
            du_stack_d[__idx_du_stack_0] = dud[i_x]
            du_stack[__idx_du_stack_0] = du[i_x]
            dud[i_x] = ud[i_x] + -(ud[i_x - 1])
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            __hcse_0 = du[i_x]
            __hcse_1 = c * dt * __hcse_0
            ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * __hcse_0) * cd + (c * __hcse_0) * dtd) + (c * dt) * dud[i_x]) + -(__hcse_1 / dx ^ 2) * dxd))
            u[i_x] = u[i_x] - __hcse_1 / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x = i_nnode:-1:2
            __cse_0d = dud[i_x]
            __cse_0 = du[i_x]
            __cse_1d = -(ubd[i_x])
            __cse_1 = -(ub[i_x])
            __hcse_2 = dx ^ 2
            __hcse_3 = 1.0 / __hcse_2
            __hcse_4 = 1.0 / dx
            __cse_2d = __cse_1 * (-__hcse_3 * dxd) + __hcse_4 * __cse_1d
            __cse_2 = __hcse_4 * __cse_1
            __hcse_5 = dt * __cse_0
            cbd = cbd + (__cse_2 * (__cse_0 * dtd + dt * __cse_0d) + __hcse_5 * __cse_2d)
            cb = cb + __hcse_5 * __cse_2
            __hcse_6 = c * __cse_0
            dtbd = dtbd + (__cse_2 * (__cse_0 * cd + c * __cse_0d) + __hcse_6 * __cse_2d)
            dtb = dtb + __hcse_6 * __cse_2
            __hcse_7 = c * dt
            dubd[i_x] = dubd[i_x] + (__cse_2 * (dt * cd + c * dtd) + __hcse_7 * __cse_2d)
            dub[i_x] = dub[i_x] + __hcse_7 * __cse_2
            __hcse_8 = c * dt * __cse_0
            __hcse_9 = -(__hcse_8 / __hcse_2)
            dxbd = dxbd + (__cse_1 * -((__hcse_3 * ((__hcse_5 * cd + __hcse_6 * dtd) + __hcse_7 * __cse_0d) + -(__hcse_8 / __hcse_2 ^ 2) * ((2dx) * dxd))) + __hcse_9 * __cse_1d)
            dxb = dxb + __hcse_9 * __cse_1
        end
        for i_x = i_nnode:-1:2
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
            dud[i_x] = du_stack_d[__idx_du_stack_0]
            du[i_x] = du_stack[__idx_du_stack_0]
            __oldb_2d = dubd[i_x]
            __oldb_2 = dub[i_x]
            dubd[i_x] = 0.0
            dub[i_x] = 0.0
            ubd[i_x] = ubd[i_x] + __oldb_2d
            ub[i_x] = ub[i_x] + __oldb_2
            ubd[i_x - 1] = ubd[i_x - 1] + -__oldb_2d
            ub[i_x - 1] = ub[i_x - 1] + -__oldb_2
        end
    end
    return (cb, cbd, dxb, dxbd, dtb, dtbd)
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
