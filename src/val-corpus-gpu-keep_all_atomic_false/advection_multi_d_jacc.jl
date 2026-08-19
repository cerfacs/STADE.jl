import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_advection_multi_d_1!(__jacc_i, du, dud, i_nnode, u, ud)
    i_x_advection_diff_c1 = 2 + (__jacc_i - 1)
    dud[i_x_advection_diff_c1] = ud[i_x_advection_diff_c1] + -(ud[i_x_advection_diff_c1 - 1])
    du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
    return nothing
end

function jacc_kernel_advection_multi_d_2!(__jacc_i, c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    i_x_advection_update_c2 = 2 + (__jacc_i - 1)
    ud[i_x_advection_update_c2] = ud[i_x_advection_update_c2] + -(((1.0 / dx) * (((dt * du[i_x_advection_update_c2]) * cd + (c * du[i_x_advection_update_c2]) * dtd) + (c * dt) * dud[i_x_advection_update_c2]) + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * dxd))
    u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
    return nothing
end

function jacc_kernel_advection_multi_1!(__jacc_i, du, i_nnode, u)
    i_x_advection_diff_c1 = 2 + (__jacc_i - 1)
    du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
    return nothing
end

function jacc_kernel_advection_multi_2!(__jacc_i, c, dt, du, dx, i_nnode, u)
    i_x_advection_update_c2 = 2 + (__jacc_i - 1)
    u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
    return nothing
end

function advection_multi_d_jacc(u, ud, du, dud, c, cd, dx, dxd, dt, dtd, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_d_1!(du, dud, i_nnode, u, ud)
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_d_2!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    end
    return nothing
end

function advection_multi_jacc(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_1!(du, i_nnode, u)
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
