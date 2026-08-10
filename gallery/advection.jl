# Function that advances in time the state variable `u` following an upwind discretization of the 1D advection equation. Variables `c`, `dx`, `dt`, `i_nstep`, and `i_nnode` stand for the advection speed, the grid cell size, the time step, the number of simulation time steps, and the number of grid nodes, respectively. The variable `du` stores a finite difference inside an iteration-independent loop over the grid nodes.
function func(u, du, c, dx, dt, i_nstep, i_nnode)

  # Loop over time steps.
  # Since the loop variable starts with `i_seq_`,
  # this loop will be treated by JADE as sequential.
  for i_seq_ = 1:i_nstep

    # (Iteration-independent) loop over grid nodes.
    for i_x = 2:i_nnode
      du[i_x] = u[i_x] - u[i_x - 1]
    end

    # (Iteration-independent) loop over grid nodes.
    for i_x = 2:i_nnode
      u[i_x] = u[i_x] - c * dt * du[i_x] / dx
    end

  end

end
