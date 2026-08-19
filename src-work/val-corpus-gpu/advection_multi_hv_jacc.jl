import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_advection_multi_hv_1!(__jacc_i, du, du_stack, du_stack_d, dud, i_nnode, i_seq_, u, ud)
    i_x_advection_diff_c1 = 2 + (__jacc_i - 1)
    du_stack_d[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1] = dud[i_x_advection_diff_c1]
    du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1] = du[i_x_advection_diff_c1]
    dud[i_x_advection_diff_c1] = ud[i_x_advection_diff_c1] + -(ud[i_x_advection_diff_c1 - 1])
    du[i_x_advection_diff_c1] = u[i_x_advection_diff_c1] - u[i_x_advection_diff_c1 - 1]
    return nothing
end

function jacc_kernel_advection_multi_hv_2!(__jacc_i, c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    i_x_advection_update_c2 = 2 + (__jacc_i - 1)
    ud[i_x_advection_update_c2] = ud[i_x_advection_update_c2] + -(((1.0 / dx) * (((dt * du[i_x_advection_update_c2]) * cd + (c * du[i_x_advection_update_c2]) * dtd) + (c * dt) * dud[i_x_advection_update_c2]) + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * dxd))
    u[i_x_advection_update_c2] = u[i_x_advection_update_c2] - (c * dt * du[i_x_advection_update_c2]) / dx
    return nothing
end

function jacc_kernel_advection_multi_hv_3!(__jacc_i, c, cb, cbd, cd, dt, dtb, dtbd, dtd, du, dub, dubd, dud, dx, dxb, dxbd, dxd, i_nnode, ub, ubd)
    i_x_advection_update_c2 = 2 + (__jacc_i - 1)
    Atomix.@atomic cbd[1] += ((1.0 / dx) * -(ub[i_x_advection_update_c2])) * (du[i_x_advection_update_c2] * dtd + dt * dud[i_x_advection_update_c2]) + (dt * du[i_x_advection_update_c2]) * (-(ub[i_x_advection_update_c2]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x_advection_update_c2]))
    Atomix.@atomic cb[1] += (dt * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    Atomix.@atomic dtbd[1] += ((1.0 / dx) * -(ub[i_x_advection_update_c2])) * (du[i_x_advection_update_c2] * cd + c * dud[i_x_advection_update_c2]) + (c * du[i_x_advection_update_c2]) * (-(ub[i_x_advection_update_c2]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x_advection_update_c2]))
    Atomix.@atomic dtb[1] += (c * du[i_x_advection_update_c2]) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    dubd[i_x_advection_update_c2] = dubd[i_x_advection_update_c2] + (((1.0 / dx) * -(ub[i_x_advection_update_c2])) * (dt * cd + c * dtd) + (c * dt) * (-(ub[i_x_advection_update_c2]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x_advection_update_c2])))
    dub[i_x_advection_update_c2] = dub[i_x_advection_update_c2] + (c * dt) * ((1.0 / dx) * -(ub[i_x_advection_update_c2]))
    Atomix.@atomic dxbd[1] += -(ub[i_x_advection_update_c2]) * -(((1.0 / dx ^ 2) * (((dt * du[i_x_advection_update_c2]) * cd + (c * du[i_x_advection_update_c2]) * dtd) + (c * dt) * dud[i_x_advection_update_c2]) + -((c * dt * du[i_x_advection_update_c2]) / (dx ^ 2) ^ 2) * ((2dx) * dxd))) + -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * -(ubd[i_x_advection_update_c2])
    Atomix.@atomic dxb[1] += -((c * dt * du[i_x_advection_update_c2]) / dx ^ 2) * -(ub[i_x_advection_update_c2])
    return nothing
end

function jacc_kernel_advection_multi_hv_4!(__jacc_i, du, du_stack, du_stack_d, dub, dubd, dud, i_nnode, i_seq_, ub, ubd)
    i_x_advection_diff_c1 = i_nnode + (__jacc_i - 1) * -1
    dud[i_x_advection_diff_c1] = du_stack_d[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1]
    du[i_x_advection_diff_c1] = du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x_advection_diff_c1 - 2)) + 1]
    ubd[i_x_advection_diff_c1] = ubd[i_x_advection_diff_c1] + dubd[i_x_advection_diff_c1]
    ub[i_x_advection_diff_c1] = ub[i_x_advection_diff_c1] + dub[i_x_advection_diff_c1]
    ubd[i_x_advection_diff_c1 - 1] = ubd[i_x_advection_diff_c1 - 1] + -(dubd[i_x_advection_diff_c1])
    ub[i_x_advection_diff_c1 - 1] = ub[i_x_advection_diff_c1 - 1] + -(dub[i_x_advection_diff_c1])
    dubd[i_x_advection_diff_c1] = 0.0
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

function advection_multi_hv_jacc(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    cb = JACC.array([cb])
    cbd = JACC.array([cbd])
    dtb = JACC.array([dtb])
    dtbd = JACC.array([dtbd])
    dxb = JACC.array([dxb])
    dxbd = JACC.array([dxbd])
    du_stack_d = JACC.zeros(Float64, length(du_stack))
    for i_seq_ = 1:i_nstep
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_hv_1!(du, du_stack, du_stack_d, dud, i_nnode, i_seq_, u, ud)
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_hv_2!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    end
    for i_seq_ = i_nstep:-1:1
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_hv_3!(c, cb, cbd, cd, dt, dtb, dtbd, dtd, du, dub, dubd, dud, dx, dxb, dxbd, dxd, i_nnode, ub, ubd)
        JACC.@parallel_for range = div(2 - i_nnode, -1) + 1 jacc_kernel_advection_multi_hv_4!(du, du_stack, du_stack_d, dub, dubd, dud, i_nnode, i_seq_, ub, ubd)
    end
    cb = (JACC.to_host(cb))[1]
    cbd = (JACC.to_host(cbd))[1]
    dtb = (JACC.to_host(dtb))[1]
    dtbd = (JACC.to_host(dtbd))[1]
    dxb = (JACC.to_host(dxb))[1]
    dxbd = (JACC.to_host(dxbd))[1]
    return (cb, cbd, dxb, dxbd, dtb, dtbd)
end

function advection_multi_jacc(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_1!(du, i_nnode, u)
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_multi_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
