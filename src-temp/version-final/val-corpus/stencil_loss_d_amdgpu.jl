import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_stencil_loss_d_1!(i_n, u, ud, w, wd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function amdgpu_kernel_stencil_loss_1!(i_n, u, w)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function stencil_loss_d_amdgpu(loss, lossd, u, ud, w, wd, i_n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) amdgpu_kernel_stencil_loss_d_1!(i_n, u, ud, w, wd)
    for i_seq_x = 2:i_n - 1
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end

function stencil_loss_amdgpu(loss, u, w, i_n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) amdgpu_kernel_stencil_loss_1!(i_n, u, w)
    for i_seq_x = 2:i_n - 1
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end
