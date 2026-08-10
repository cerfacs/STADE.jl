import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

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

function advection_amdgpu(u, du, c, dx, dt, i_nstep, i_nnode)
    nthread_per_block = 256
    for i_seq_ = 1:i_nstep
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_1!(du, i_nnode, u)
        @roc groupsize = nthread_per_block gridsize = cld(div(i_nnode - 2, 1) + 1, nthread_per_block) amdgpu_kernel_advection_2!(c, dt, du, dx, i_nnode, u)
    end
    return nothing
end
