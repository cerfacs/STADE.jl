import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_advection_b_1!(c, dt, du, dx, i_nnode, u)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    u[i_x] = u[i_x] - (c * dt * du[i_x]) / dx
    return nothing
end

function amdgpu_kernel_advection_b_2!(c, dt, du, dub, dx, i_nnode, ub)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_nnode - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    cb = cb + (dt * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
    dtb = dtb + (c * du[i_x]) * ((1.0 / dx) * -(ub[i_x]))
    dub[i_x] = dub[i_x] + (c * dt) * ((1.0 / dx) * -(ub[i_x]))
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

function advection_b_amdgpu(u, ub, du, dub, c, cb, dx, dxb, dt, dtb, i_nstep, i_nnode, du_stack)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        for i_x = 2:i_nnode
            push!(du_stack, du[i_x])
            du[i_x] = u[i_x] - u[i_x - 1]
        end
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_b_1!(c, dt, du, dx, i_nnode, u)
    end
    for i_seq_ = i_nstep:-1:1
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_b_2!(c, dt, du, dub, dx, i_nnode, ub)
        for i_x = i_nnode:-1:2
            du[i_x] = pop!(du_stack)
            ub[i_x] = ub[i_x] + dub[i_x]
            ub[i_x - 1] = ub[i_x - 1] + -(dub[i_x])
            dub[i_x] = 0.0
        end
    end
    return (cb, dxb, dtb)
end

function advection_amdgpu(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_1!(du, i_nnode, u)
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
