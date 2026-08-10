import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_affine_loss_hv_1!(a, ad, b, bd, i_n, u, ud, v, vd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function amdgpu_kernel_affine_loss_hv_2!(a, ab, abd, ad, bb, bbd, i_n, u, ub, ubd, ud, vb, vbd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    abd[i_x] = abd[i_x] + (vb[i_x] * ud[i_x] + u[i_x] * vbd[i_x])
    ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
    ubd[i_x] = ubd[i_x] + (vb[i_x] * ad[i_x] + a[i_x] * vbd[i_x])
    ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
    bbd[i_x] = bbd[i_x] + vbd[i_x]
    bb[i_x] = bb[i_x] + vb[i_x]
    vbd[i_x] = 0.0
    vb[i_x] = 0.0
    return nothing
end

function amdgpu_kernel_affine_loss_1!(a, b, i_n, u, v)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function initstacks_affine_loss_b_amdgpu()
    return nothing
end

function affine_loss_hv_amdgpu(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, bd, bbd, vd, vbd)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_affine_loss_hv_1!(a, ad, b, bd, i_n, u, ud, v, vd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * v[i_seq_x]) * vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        vbd[i_seq_x] = vbd[i_seq_x] + (lossb[1] * (2 * vd[i_seq_x]) + (2 * v[i_seq_x]) * lossbd[1])
        vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    end
    @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_affine_loss_hv_2!(a, ab, abd, ad, bb, bbd, i_n, u, ub, ubd, ud, vb, vbd)
    return nothing
end

function affine_loss_amdgpu(loss, u, a, b, v, i_n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_affine_loss_1!(a, b, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    return nothing
end
