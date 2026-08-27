function initstacks_advection_multi_b(i_nnode, i_nstep)
    du_stack = Vector{Float64}(undef, max(0, div(i_nstep - 1, 1) + 1) * max(0, div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_multi_b(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    for i_ = 1:i_nstep
        for i_x_advection_diff_c1 = 2:i_nnode
            du_stack[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1] = du[i_x_advection_diff_c1]
            du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
        end
        for i_x_advection_update_c2 = 2:i_nnode
            u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
        end
    end
    for i_ = i_nstep:-1:1
        for i_x_advection_update_c2 = i_nnode:-1:2
            cb = cb + (dt * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dtb = dtb + (c * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dub[i_x_advection_update_c2] = dub[i_x_advection_update_c2] + (c * dt) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dxb = dxb + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * -(ub[i_x_advection_update_c2])
        end
        for i_x_advection_diff_c1 = i_nnode:-1:2
            du[i_x_advection_diff_c1] = du_stack[((i_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1]
            ub[i_x_advection_diff_c1] = ub[i_x_advection_diff_c1] + dub[i_x_advection_diff_c1]
            ub[i_x_advection_diff_c1 - 1] = ub[i_x_advection_diff_c1 - 1] + -(dub[i_x_advection_diff_c1])
            dub[i_x_advection_diff_c1] = 0.0
        end
    end
    return (cb, dxb, dtb)
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
