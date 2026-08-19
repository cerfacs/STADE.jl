import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_advection_multi_b_1!(__jacc_i, du, du_stack, i_nnode, i_seq_, u)
    i_x_advection_diff_c1 = 2 + (__jacc_i - 1)
    du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1] = du[i_x_advection_diff_c1]
    du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
    return nothing
end

function jacc_kernel_advection_multi_b_2!(__jacc_i, c, dt, du, dx, i_nnode, u)
    i_x_advection_update_c2 = 2 + (__jacc_i - 1)
    u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
    return nothing
end

function jacc_kernel_advection_multi_b_3!(__jacc_i, c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
    i_x_advection_update_c2 = 2 + (__jacc_i - 1)
    Atomix.@atomic cb[1] += (dt * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    Atomix.@atomic dtb[1] += (c * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    dub[i_x_advection_update_c2] = dub[i_x_advection_update_c2] + (c * dt) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    Atomix.@atomic dxb[1] += -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * -(ub[i_x_advection_update_c2])
    return nothing
end

function jacc_kernel_advection_multi_b_4!(__jacc_i, du, du_stack, dub, i_nnode, i_seq_, ub)
    i_x_advection_diff_c1 = i_nnode + (__jacc_i - 1) * -1
    du[i_x_advection_diff_c1] = du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1]
    ub[i_x_advection_diff_c1] = ub[i_x_advection_diff_c1] + dub[i_x_advection_diff_c1]
    ub[i_x_advection_diff_c1 - 1] = ub[i_x_advection_diff_c1 - 1] + -(dub[i_x_advection_diff_c1])
    dub[i_x_advection_diff_c1] = 0.0
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

function initstacks_advection_multi_b_jacc(i_nnode, i_nstep)
    du_stack = JACC.zeros(Float64, (div(i_nstep - 1, 1) + 1) * (div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_multi_b_jacc(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    cb = JACC.array([cb])
    dtb = JACC.array([dtb])
    dxb = JACC.array([dxb])
    for i_seq_ = 1:i_nstep
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_b_1!(du, du_stack, i_nnode, i_seq_, u)
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_b_2!(c, dt, du, dx, i_nnode, u)
    end
    for i_seq_ = i_nstep:-1:1
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_b_3!(c, cb, dt, dtb, du, dub, dx, dxb, i_nnode, ub)
        JACC.@parallel_for range = div(2 - i_nnode, -1) + 1 jacc_kernel_advection_multi_b_4!(du, du_stack, dub, i_nnode, i_seq_, ub)
    end
    cb = (JACC.to_host(cb))[1]
    dtb = (JACC.to_host(dtb))[1]
    dxb = (JACC.to_host(dxb))[1]
    return (cb, dxb, dtb)
end

function advection_multi_jacc(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_1!(du, i_nnode, u)
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
