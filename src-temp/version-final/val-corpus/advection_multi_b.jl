function initstacks_advection_diff_b()
    return nothing
end

function advection_diff_b(u, ub, du, dub, i_nnode)
    for i_x = 2:i_nnode
        du[i_x] = u[i_x] - u[i_x - 1]
    end
    for i_x = 2:i_nnode
        ub[i_x] = ub[i_x] + dub[i_x]
        ub[i_x - 1] = ub[i_x - 1] + -(dub[i_x])
        dub[i_x] = 0.0
    end
    return nothing
end

function advection_diff(u, du, i_nnode)
    #= none:1 =#
    #= none:2 =#
    for i_x = 2:i_nnode
        #= none:3 =#
        du[i_x] = u[i_x] - u[i_x - 1]
        #= none:4 =#
    end
end

function initstacks_advection_multi_b()
    du_stack = Vector{Float64}()
    return du_stack
end

function advection_multi_b(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    for i_seq_ = 1:i_nstep
        for i_x_advection_diff_c1 = 2:i_nnode
            push!(du_stack, du[i_x_advection_diff_c1])
            du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
        end
        for i_x_advection_update_c2 = 2:i_nnode
            u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
        end
    end
    for i_seq_ = i_nstep:-1:1
        for i_x_advection_update_c2 = 2:i_nnode
            cb = cb + (dt * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dtb = dtb + (c * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dub[i_x_advection_update_c2] = dub[i_x_advection_update_c2] + (c * dt) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
            dxb = dxb + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * -(ub[i_x_advection_update_c2])
        end
        for i_x_advection_diff_c1 = i_nnode:-1:2
            du[i_x_advection_diff_c1] = pop!(du_stack)
            ub[i_x_advection_diff_c1] = ub[i_x_advection_diff_c1] + dub[i_x_advection_diff_c1]
            ub[i_x_advection_diff_c1 - 1] = ub[i_x_advection_diff_c1 - 1] + -(dub[i_x_advection_diff_c1])
            dub[i_x_advection_diff_c1] = 0.0
        end
    end
    return (cb, dxb, dtb)
end

function advection_multi(u, du, c, dx, dt, i_nstep, i_nnode)
    #= none:13 =#
    #= none:14 =#
    for i_seq_ = 1:i_nstep
        #= none:15 =#
        advection_diff(u, du, i_nnode)
        #= none:16 =#
        advection_update(u, du, c, dx, dt, i_nnode)
        #= none:17 =#
    end
end

function initstacks_advection_update_b()
    return nothing
end

function advection_update_b(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nnode)
    for i_x = 2:i_nnode
        u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    end
    for i_x = 2:i_nnode
        cb = cb + (dt * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
        dtb = dtb + (c * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
        dub[i_x] = dub[i_x] + (c * dt) * ((1.0 / dx) * -(ub[i_x]))
        dxb = dxb + -((c * dt * du[i_x]) / dx ^ 2) * -(ub[i_x])
    end
    return (cb, dxb, dtb)
end

function advection_update(u, du, c, dx, dt, i_nnode)
    #= none:7 =#
    #= none:8 =#
    for i_x = 2:i_nnode
        #= none:9 =#
        u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        #= none:10 =#
    end
end
