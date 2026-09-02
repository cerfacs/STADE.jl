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
            __hcse_0 = du[i_x_advection_update_c2]
            __hcse_1 = c * dt * __hcse_0
            ud[i_x_advection_update_c2] = ud[i_x_advection_update_c2] + -(((1.0 / dx) * (((dt * __hcse_0) * cd + (c * __hcse_0) * dtd) + (c * dt) * dud[i_x_advection_update_c2]) + -(__hcse_1 / dx ^ 2) * dxd))
            u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - __hcse_1 / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x_advection_update_c2 = i_nnode:-1:2
            __cse_0d = dud[i_x_advection_update_c2]
            __cse_0 = du[i_x_advection_update_c2]
            __cse_1d = -(ubd[i_x_advection_update_c2])
            __cse_1 = -(ub[i_x_advection_update_c2])
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
            dubd[i_x_advection_update_c2] = dubd[i_x_advection_update_c2] + (__cse_2 * (dt * cd + c * dtd) + __hcse_7 * __cse_2d)
            dub[i_x_advection_update_c2] = dub[i_x_advection_update_c2] + __hcse_7 * __cse_2
            __hcse_8 = c * dt * __cse_0
            __hcse_9 = -(__hcse_8 / __hcse_2)
            dxbd = dxbd + (__cse_1 * -((__hcse_3 * ((__hcse_5 * cd + __hcse_6 * dtd) + __hcse_7 * __cse_0d) + -(__hcse_8 / __hcse_2 ^ 2) * ((2dx) * dxd))) + __hcse_9 * __cse_1d)
            dxb = dxb + __hcse_9 * __cse_1
        end
        for i_x_advection_diff_c1 = i_nnode:-1:2
            __idx_du_stack_0 = ((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1
            dud[i_x_advection_diff_c1] = du_stack_d[__idx_du_stack_0]
            du[i_x_advection_diff_c1] = du_stack[__idx_du_stack_0]
            __oldb_2d = dubd[i_x_advection_diff_c1]
            __oldb_2 = dub[i_x_advection_diff_c1]
            dubd[i_x_advection_diff_c1] = 0.0
            dub[i_x_advection_diff_c1] = 0.0
            ubd[i_x_advection_diff_c1] = ubd[i_x_advection_diff_c1] + __oldb_2d
            ub[i_x_advection_diff_c1] = ub[i_x_advection_diff_c1] + __oldb_2
            ubd[i_x_advection_diff_c1 - 1] = ubd[i_x_advection_diff_c1 - 1] + -__oldb_2d
            ub[i_x_advection_diff_c1 - 1] = ub[i_x_advection_diff_c1 - 1] + -__oldb_2
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
