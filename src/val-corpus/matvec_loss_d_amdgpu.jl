import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_matvec_loss_d_1!(a, ad, i_m, i_n, u, ud, v, vd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        vd[i_i] = vd[i_i] + (u[i_seq_j] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * ud[i_seq_j])
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function amdgpu_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function matvec_loss_d_amdgpu(loss, lossd, a, ad, u, ud, v, vd, i_m, i_n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div(i_m - 1, 1) + 1, nthread_per_block) amdgpu_kernel_matvec_loss_d_1!(a, ad, i_m, i_n, u, ud, v, vd)
    for i_seq_i = 1:i_m
        lossd[1] = lossd[1] + (2 * v[i_seq_i]) * vd[i_seq_i]
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
    return nothing
end

function matvec_loss_amdgpu(loss, a, u, v, i_m, i_n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div(i_m - 1, 1) + 1, nthread_per_block) amdgpu_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    for i_seq_i = 1:i_m
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
    return nothing
end
