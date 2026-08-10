function advection_diff_d(u, ud, du, dud, i_nnode)
    for i_x = 2:i_nnode
        dud[i_x] = ud[i_x] + -(ud[i_x - 1])
        du[i_x] = u[i_x] - u[i_x - 1]
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

function advection_multi_d(u, ud, du, dud, c, cd, dx, dxd, dt, dtd, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        for i_x_advection_diff_c1 = 2:i_nnode
            dud[i_x_advection_diff_c1] = ud[i_x_advection_diff_c1] + -(ud[i_x_advection_diff_c1 - 1])
            du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
        end
        for i_x_advection_update_c2 = 2:i_nnode
            ud[i_x_advection_update_c2] = ud[i_x_advection_update_c2] + -(((1.0 / dx) * (((dt * du[i_x_advection_update_c2]) * cd + (c * du[i_x_advection_update_c2]) * dtd) + (c * dt) * dud[i_x_advection_update_c2]) + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * dxd))
            u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
        end
    end
    return nothing
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

function advection_update_d(u, ud, du, dud, c, cd, dx, dxd, dt, dtd, i_nnode)
    for i_x = 2:i_nnode
        ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / dx ^ 2) * dxd))
        u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    end
    return nothing
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
