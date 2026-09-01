function advection_d(u, ud, du, dud, c, cd, dx, dxd, dt, dtd, i_nstep, i_nnode)
    for i_ = 1:i_nstep
        for i_x = 2:i_nnode
            dud[i_x] = ud[i_x] + -(ud[i_x - 1])
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            __cse_0 = du[i_x]
            __cse_1 = c * dt * __cse_0
            ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * __cse_0) * cd + (c * __cse_0) * dtd) + (c * dt) * dud[i_x]) + -(__cse_1 / dx ^ 2) * dxd))
            u[i_x] = u[i_x] - __cse_1 / dx
        end
    end
    return nothing
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
