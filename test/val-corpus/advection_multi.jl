function advection_diff(u, du, i_nnode)
    for i_x = 2:i_nnode
        du[i_x] = u[i_x] - u[i_x - 1]
    end
end

function advection_update(u, du, c, dx, dt, i_nnode)
    for i_x = 2:i_nnode
        u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    end
end

function advection_multi(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_ = 1:i_nstep
        advection_diff(u, du, i_nnode)
        advection_update(u, du, c, dx, dt, i_nnode)
    end
end