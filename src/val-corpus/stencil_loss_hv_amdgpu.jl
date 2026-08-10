import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_stencil_loss_hv_1!(i_n, u, ud, w, wd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function amdgpu_kernel_stencil_loss_hv_2!(i_n, ub, ubd, wb, wbd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    ubd[i_x - 1] = ubd[i_x - 1] + wbd[i_x]
    ub[i_x - 1] = ub[i_x - 1] + wb[i_x]
    ubd[i_x] = ubd[i_x] + 2.0 * -(wbd[i_x])
    ub[i_x] = ub[i_x] + 2.0 * -(wb[i_x])
    ubd[i_x + 1] = ubd[i_x + 1] + wbd[i_x]
    ub[i_x + 1] = ub[i_x + 1] + wb[i_x]
    wbd[i_x] = 0.0
    wb[i_x] = 0.0
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

function initstacks_stencil_loss_b_amdgpu()
    return nothing
end

function stencil_loss_hv_amdgpu(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) amdgpu_kernel_stencil_loss_hv_1!(i_n, u, ud, w, wd)
    for i_seq_x = 2:i_n - 1
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    for i_seq_x = i_n - 1:-1:2
        wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * (2 * wd[i_seq_x]) + (2 * w[i_seq_x]) * lossbd[1])
        wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    end
    @roc groupsize = nthread_per_block gridsize = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) amdgpu_kernel_stencil_loss_hv_2!(i_n, ub, ubd, wb, wbd)
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
