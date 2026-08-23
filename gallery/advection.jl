# advection(u, du, c, dx, dt, i_nstep, i_nnode)
#
# i_nstep explicit first-order upwind time steps of the 1D linear
# advection equation: each step differences u backward onto du, then
# advances u by -c*dt*du/dx. Node 1 is left untouched (no boundary
# condition handling).
#
# u: field value at each node, length i_nnode, updated in place
# du: scratch array for the backward difference u[i]-u[i-1], length i_nnode
# c: advection speed
# dx: grid spacing
# dt: time step size
# i_nstep: number of time steps to take
# i_nnode: number of nodes
function advection(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
    return nothing
end

function advection(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        for i_x = 2:i_nnode
            u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
        end
    end
end