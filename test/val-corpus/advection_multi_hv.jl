function initstacks_advection_multi_b(i_nnode, i_nstep)
    du_stack = Vector{Float64}(undef, (div(i_nstep - 1, 1) + 1) * (div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_multi_hv(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    du_stack_d = Vector{Float64}(undef, length(du_stack))
    for i_ = 1:i_nstep
        for i_x_advection_diff_c1 = 2:i_nnode
            du_stack_d[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1] = dud[i_x_advection_diff_c1]
            du_stack[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1] = du[i_x_advection_diff_c1]
            dud[i_x_advection_diff_c1] = ud[i_x_advection_diff_c1] + -(ud[i_x_advection_diff_c1 - 1])
            du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
        end
        for i_x_advection_update_c2 = 2:i_nnode
            ud[i_x_advection_update_c2] = ud[i_x_advection_update_c2] + -(((1.0 / dx) * (((dt * du[i_x_advection_update_c2]) * cd + (c * du[i_x_advection_update_c2]) * dtd) + (c * dt) * dud[i_x_advection_update_c2]) + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * dxd))
            u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x_advection_update_c2 = i_nnode:-1:2
            cbd = cbd + (((1.0 / dx) * -(ub[i_x_advection_update_c2])) * (du[i_x_advection_update_c2] * dtd + dt * dud[i_x_advection_update_c2]) + (dt * du[i_x_advection_update_c2]) * (-(ub[i_x_advection_update_c2]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x_advection_update_c2])))
            cb = cb + (dt * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dtbd = dtbd + (((1.0 / dx) * -(ub[i_x_advection_update_c2])) * (du[i_x_advection_update_c2] * cd + c * dud[i_x_advection_update_c2]) + (c * du[i_x_advection_update_c2]) * (-(ub[i_x_advection_update_c2]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x_advection_update_c2])))
            dtb = dtb + (c * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dubd[i_x_advection_update_c2] = dubd[i_x_advection_update_c2] + (((1.0 / dx) * -(ub[i_x_advection_update_c2])) * (dt * cd + c * dtd) + (c * dt) * (-(ub[i_x_advection_update_c2]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x_advection_update_c2])))
            dub[i_x_advection_update_c2] = dub[i_x_advection_update_c2] + (c * dt) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dxbd = dxbd + (-(ub[i_x_advection_update_c2]) * -(((1.0 / dx ^ 2) * (((dt * du[i_x_advection_update_c2]) * cd + (c * du[i_x_advection_update_c2]) * dtd) + (c * dt) * dud[i_x_advection_update_c2]) + -((c * dt * du[i_x_advection_update_c2]) / (dx ^ 2) ^ 2) * ((2dx) * dxd))) + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * -(ubd[i_x_advection_update_c2]))
            dxb = dxb + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * -(ub[i_x_advection_update_c2])
        end
        for i_x_advection_diff_c1 = i_nnode:-1:2
            dud[i_x_advection_diff_c1] = du_stack_d[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1]
            du[i_x_advection_diff_c1] = du_stack[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1]
            ubd[i_x_advection_diff_c1] = ubd[i_x_advection_diff_c1] + dubd[i_x_advection_diff_c1]
            ub[i_x_advection_diff_c1] = ub[i_x_advection_diff_c1] + dub[i_x_advection_diff_c1]
            ubd[i_x_advection_diff_c1 - 1] = ubd[i_x_advection_diff_c1 - 1] + -(dubd[i_x_advection_diff_c1])
            ub[i_x_advection_diff_c1 - 1] = ub[i_x_advection_diff_c1 - 1] + -(dub[i_x_advection_diff_c1])
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
