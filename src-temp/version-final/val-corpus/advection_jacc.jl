import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_advection_1!(__jacc_i, du, i_nnode, u)
    i_x = 2 + (__jacc_i - 1)
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function jacc_kernel_advection_2!(__jacc_i, c, dt, du, dx, i_nnode, u)
    i_x = 2 + (__jacc_i - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function advection_jacc(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        JACC.parallel_for(div(i_nnode - 2, 1) + 1, jacc_kernel_advection_1!, du, i_nnode, u)
        JACC.parallel_for(div(i_nnode - 2, 1) + 1, jacc_kernel_advection_2!, c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
