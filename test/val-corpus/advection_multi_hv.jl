function initstacks_advection_multi_b(i_nnode, i_nstep)
    du_stack = Vector{Float64}(undef, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_multi_hv(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    du_stack_d = Vector{Float64}(undef, length(du_stack))
    for i_ = 1:i_nstep
        for i_x_advection_diff_c1 = 2:i_nnode
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1
            du_stack_d[__idx_du_stack_0] = dud[i_x_advection_diff_c1]
            du_stack[__idx_du_stack_0] = du[i_x_advection_diff_c1]
            dud[i_x_advection_diff_c1] = ud[i_x_advection_diff_c1] + -(ud[i_x_advection_diff_c1 - 1])
            du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
        end
        for i_x_advection_update_c2 = 2:i_nnode
            __cse_4 = du[i_x_advection_update_c2]
            __cse_5 = c * dt * __cse_4
            ud[i_x_advection_update_c2] = ud[i_x_advection_update_c2] + -(((1.0 / dx) * (((dt * __cse_4) * cd + (c * __cse_4) * dtd) + (c * dt) * dud[i_x_advection_update_c2]) + -(__cse_5 / dx ^ 2) * dxd))
            u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - __cse_5 / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x_advection_update_c2 = i_nnode:-1:2
            __cse_0d = dud[i_x_advection_update_c2]
            __cse_0 = du[i_x_advection_update_c2]
            __cse_1d = -(ubd[i_x_advection_update_c2])
            __cse_1 = -(ub[i_x_advection_update_c2])
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
            dubd[i_x_advection_update_c2] = dubd[i_x_advection_update_c2] + (__cse_2 * (dt * cd + c * dtd) + __cse_11 * __cse_2d)
            dub[i_x_advection_update_c2] = dub[i_x_advection_update_c2] + __cse_11 * __cse_2
            __cse_12 = c * dt * __cse_0
            __cse_13 = -(__cse_12 / __cse_6)
            dxbd = dxbd + (__cse_1 * -((__cse_7 * ((__cse_9 * cd + __cse_10 * dtd) + __cse_11 * __cse_0d) + -(__cse_12 / __cse_6 ^ 2) * ((2dx) * dxd))) + __cse_13 * __cse_1d)
            dxb = dxb + __cse_13 * __cse_1
        end
        for i_x_advection_diff_c1 = i_nnode:-1:2
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1
            dud[i_x_advection_diff_c1] = du_stack_d[__idx_du_stack_0]
            du[i_x_advection_diff_c1] = du_stack[__idx_du_stack_0]
            __cse_3d = dubd[i_x_advection_diff_c1]
            __cse_3 = dub[i_x_advection_diff_c1]
            ubd[i_x_advection_diff_c1] = ubd[i_x_advection_diff_c1] + __cse_3d
            ub[i_x_advection_diff_c1] = ub[i_x_advection_diff_c1] + __cse_3
            ubd[i_x_advection_diff_c1 - 1] = ubd[i_x_advection_diff_c1 - 1] + -__cse_3d
            ub[i_x_advection_diff_c1 - 1] = ub[i_x_advection_diff_c1 - 1] + -__cse_3
            dubd[i_x_advection_diff_c1] = 0.0
            dub[i_x_advection_diff_c1] = 0.0
        end
    end
    return (cb, cbd, dxb, dxbd, dtb, dtbd)
end

function advection_multi(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_ = 1:i_nstep
        for i_x_advection_diff_c1 = 2:i_nnode
            du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
        end
        for i_x_advection_update_c2 = 2:i_nnode
            u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
        end
    end
end
