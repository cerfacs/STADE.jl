import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_advection_hv_1!(__jacc_i, du, du_stack, du_stack_d, dud, i_nnode, i_seq_, u, ud)
    i_x = 2 + (__jacc_i - 1)
    du_stack_d[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1] = dud[i_x]
    du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1] = du[i_x]
    dud[i_x] = ud[i_x] + -(ud[i_x - 1])
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function jacc_kernel_advection_hv_2!(__jacc_i, c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    i_x = 2 + (__jacc_i - 1)
    ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / dx ^ 2) * dxd))
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function jacc_kernel_advection_hv_3!(__jacc_i, c, cb, cbd, cd, dt, dtb, dtbd, dtd, du, dub, dubd, dud, dx, dxb, dxbd, dxd, i_nnode, ub, ubd)
    i_x = 2 + (__jacc_i - 1)
    Atomix.@atomic cbd[1] += ((1.0 / dx) * -(ub[i_x])) * (du[i_x] * dtd + dt * dud[i_x]) + (dt * du[i_x]) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x]))
    Atomix.@atomic cb[1] += (dt * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
    Atomix.@atomic dtbd[1] += ((1.0 / dx) * -(ub[i_x])) * (du[i_x] * cd + c * dud[i_x]) + (c * du[i_x]) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x]))
    Atomix.@atomic dtb[1] += (c * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
    dubd[i_x] = dubd[i_x] + (((1.0 / dx) * -(ub[i_x])) * (dt * cd + c * dtd) + (c * dt) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x])))
    dub[i_x] = dub[i_x] + (c * dt) * ((1.0 / dx) * -(ub[i_x]))
    Atomix.@atomic dxbd[1] += -(ub[i_x]) * -(((1.0 / dx ^ 2) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / (dx ^ 2) ^ 2) * ((2dx) * dxd))) + -((c * dt * du[i_x]) / dx ^ 2) * -(ubd[i_x])
    Atomix.@atomic dxb[1] += -((c * dt * du[i_x]) / dx ^ 2) * -(ub[i_x])
    return nothing
end

function jacc_kernel_advection_hv_4!(__jacc_i, du, du_stack, du_stack_d, dub, dubd, dud, i_nnode, i_seq_, ub, ubd)
    i_x = i_nnode + (__jacc_i - 1) * -1
    dud[i_x] = du_stack_d[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1]
    du[i_x] = du_stack[((i_seq_ - 1) * (div(i_nnode - 2, 1) + 1) + (i_x - 2)) + 1]
    ubd[i_x] = ubd[i_x] + dubd[i_x]
    ub[i_x] = ub[i_x] + dub[i_x]
    ubd[i_x - 1] = ubd[i_x - 1] + -(dubd[i_x])
    ub[i_x - 1] = ub[i_x - 1] + -(dub[i_x])
    dubd[i_x] = 0.0
    dub[i_x] = 0.0
    return nothing
end

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

function initstacks_advection_b_jacc(i_nnode, i_nstep)
    du_stack = JACC.zeros(Float64, (div(i_nstep - 1, 1) + 1) * (div(i_nnode - 2, 1) + 1))
    return du_stack
end

function advection_hv_jacc(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    cb = JACC.array([cb])
    cbd = JACC.array([cbd])
    dtb = JACC.array([dtb])
    dtbd = JACC.array([dtbd])
    dxb = JACC.array([dxb])
    dxbd = JACC.array([dxbd])
    du_stack_d = JACC.zeros(Float64, (div(i_nstep - 1, 1) + 1) * (div(i_nnode - 2, 1) + 1))
    for i_seq_ = 1:i_nstep
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_hv_1!(du, du_stack, du_stack_d, dud, i_nnode, i_seq_, u, ud)
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_hv_2!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    end
    for i_seq_ = i_nstep:-1:1
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_hv_3!(c, cb, cbd, cd, dt, dtb, dtbd, dtd, du, dub, dubd, dud, dx, dxb, dxbd, dxd, i_nnode, ub, ubd)
        JACC.@parallel_for range = div(2 - i_nnode, -1) + 1 jacc_kernel_advection_hv_4!(du, du_stack, du_stack_d, dub, dubd, dud, i_nnode, i_seq_, ub, ubd)
    end
    cb = (JACC.to_host(cb))[1]
    cbd = (JACC.to_host(cbd))[1]
    dtb = (JACC.to_host(dtb))[1]
    dtbd = (JACC.to_host(dtbd))[1]
    dxb = (JACC.to_host(dxb))[1]
    dxbd = (JACC.to_host(dxbd))[1]
    return (cb, cbd, dxb, dxbd, dtb, dtbd)
end

function advection_jacc(u, du, c, dx, dt, i_nstep, i_nnode)
    for i_seq_ = 1:i_nstep
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_1!(du, i_nnode, u)
        JACC.@parallel_for range = div(i_nnode - 2, 1) + 1 jacc_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
