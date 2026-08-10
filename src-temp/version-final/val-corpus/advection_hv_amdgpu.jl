import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_advection_hv_1!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    ud[i_x] = ud[i_x] + -(((1.0 / dx) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / dx ^ 2) * dxd))
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function amdgpu_kernel_advection_hv_2!(c, cd, dt, dtd, du, dub, dubd, dud, dx, dxd, i_nnode, ub, ubd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    cbd = cbd + (((1.0 / dx) * -(ub[i_x])) * (du[i_x] * dtd + dt * dud[i_x]) + (dt * du[i_x]) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x])))
    cb = cb + (dt * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
    dtbd = dtbd + (((1.0 / dx) * -(ub[i_x])) * (du[i_x] * cd + c * dud[i_x]) + (c * du[i_x]) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x])))
    dtb = dtb + (c * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
    dubd[i_x] = dubd[i_x] + (((1.0 / dx) * -(ub[i_x])) * (dt * cd + c * dtd) + (c * dt) * (-(ub[i_x]) * (-(1.0 / dx ^ 2) * dxd) + (1.0 / dx) * -(ubd[i_x])))
    dub[i_x] = dub[i_x] + (c * dt) * ((1.0 / dx) * -(ub[i_x]))
    dxbd = dxbd + (-(ub[i_x]) * -(((1.0 / dx ^ 2) * (((dt * du[i_x]) * cd + (c * du[i_x]) * dtd) + (c * dt) * dud[i_x]) + -((c * dt * du[i_x]) / (dx ^ 2) ^ 2) * ((2dx) * dxd))) + -((c * dt * du[i_x]) / dx ^ 2) * -(ubd[i_x]))
    dxb = dxb + -((c * dt * du[i_x]) / dx ^ 2) * -(ub[i_x])
    return nothing
end

function amdgpu_kernel_advection_1!(du, i_nnode, u)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    du[i_x] = u[i_x] - u[i_x - 1]
    return nothing
end

function amdgpu_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function initstacks_advection_b_amdgpu()
    du_stack = Vector{Float64}()
    return du_stack
end

function advection_hv_amdgpu(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, ud, ubd, dud, dubd, cd, cbd, dxd, dxbd, dtd, dtbd, du_stack)
    nthread_per_block = 256
    du_stack_d = Vector{Float64}()
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            push!(du_stack_d, dud[i_x])
            push!(du_stack, du[i_x])
            dud[i_x] = ud[i_x] + -(ud[i_x - 1])
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_hv_1!(c, cd, dt, dtd, du, dud, dx, dxd, i_nnode, u, ud)
    end
    for i_seq_ = i_nstep:-1:1
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_hv_2!(c, cd, dt, dtd, du, dub, dubd, dud, dx, dxd, i_nnode, ub, ubd)
        for i_x = i_nnode:-1:2
            dud[i_x] = pop!(du_stack_d)
            du[i_x] = pop!(du_stack)
            ubd[i_x] = ubd[i_x] + dubd[i_x]
            ub[i_x] = ub[i_x] + dub[i_x]
            ubd[i_x - 1] = ubd[i_x - 1] + -(dubd[i_x])
            ub[i_x - 1] = ub[i_x - 1] + -(dub[i_x])
            dubd[i_x] = 0.0
            dub[i_x] = 0.0
        end
    end
    return (cb, cbd, dxb, dxbd, dtb, dtbd)
end

function advection_amdgpu(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_1!(du, i_nnode, u)
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
