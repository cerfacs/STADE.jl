function advection_d(u, ud, du, dud, c, cd, dx, dxd, dt, dtd, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            dud[i_x] = ud[i_x] + -(ud[i_x - 1])
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / dx ^ 2) * dxd))
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
    return nothing
end

function advection(u, du, c, dx, dt, i_nstep, i_nnode)
    #= none:1 =#
    #= none:2 =#
    for i_seq_ = 1:i_nstep
        #= none:3 =#
        for i_x = 2:i_nnode
            #= none:4 =#
            du[i_x] = u[i_x] - u[i_x - 1]
            #= none:5 =#
        end
        #= none:6 =#
        for i_x = 2:i_nnode
            #= none:7 =#
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
            #= none:8 =#
        end
        #= none:9 =#
    end
end
