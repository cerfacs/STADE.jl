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
            __cse_4 = du[i_x]
            __cse_5 = c * dt * __cse_4
            ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * __cse_4) * cd + (c * __cse_4) * dtd) + (c * dt) * dud[i_x]) + -(__cse_5 / dx ^ 2) * dxd))
            u[i_x] = u[i_x] - __cse_5 / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x = i_nnode:-1:2
            __cse_0d = dud[i_x]
            __cse_0 = du[i_x]
            __cse_1d = -(ubd[i_x])
            __cse_1 = -(ub[i_x])
            __cse_6 = dx ^ 2
            __cse_7 = 1.0 / __cse_6
            __cse_8 = 1.0 / dx
            __cse_2d = __cse_1 * (-__cse_7 * dxd) + __cse_8 * __cse_1d
            __cse_2 = __cse_8 * __cse_1
            __cse_9 = dt * __cse_0
            cbd = cbd + (__cse_2 * (__cse_0 * dtd + dt * __cse_0d) + __cse_9 * __cse_2d)
            cb = cb + __cse_9 * __cse_2
            __cse_10 = c * __cse_0
            dtbd = dtbd + (__cse_2 * (__cse_0 * cd + c * __cse_0d) + __cse_10 * __cse_2d)
            dtb = dtb + __cse_10 * __cse_2
            __cse_11 = c * dt
            dubd[i_x] = dubd[i_x] + (__cse_2 * (dt * cd + c * dtd) + __cse_11 * __cse_2d)
            dub[i_x] = dub[i_x] + __cse_11 * __cse_2
            __cse_12 = c * dt * __cse_0
            __cse_13 = -(__cse_12 / __cse_6)
            dxbd = dxbd + (__cse_1 * -((__cse_7 * ((__cse_9 * cd + __cse_10 * dtd) + __cse_11 * __cse_0d) + -(__cse_12 / __cse_6 ^ 2) * ((2dx) * dxd))) + __cse_13 * __cse_1d)
            dxb = dxb + __cse_13 * __cse_1
        end
        for i_x = i_nnode:-1:2
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1
            dud[i_x] = du_stack_d[__idx_du_stack_0]
            du[i_x] = du_stack[__idx_du_stack_0]
            __cse_3d = dubd[i_x]
            __cse_3 = dub[i_x]
            ubd[i_x] = ubd[i_x] + __cse_3d
            ub[i_x] = ub[i_x] + __cse_3
            ubd[i_x - 1] = ubd[i_x - 1] + -__cse_3d
            ub[i_x - 1] = ub[i_x - 1] + -__cse_3
            dubd[i_x] = 0.0
            dub[i_x] = 0.0
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
