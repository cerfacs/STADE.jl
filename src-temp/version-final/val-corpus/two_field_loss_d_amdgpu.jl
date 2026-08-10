import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_two_field_loss_d_1!(i_n, p, pd, u, ud)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    pd[i_x] = (2 * u[i_x]) * ud[i_x]
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function amdgpu_kernel_two_field_loss_d_2!(i_n, q, qd, v, vd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    qd[i_x] = (3 * v[i_x] ^ 2) * vd[i_x]
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function amdgpu_kernel_two_field_loss_1!(i_n, p, u)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function amdgpu_kernel_two_field_loss_2!(i_n, q, v)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function two_field_loss_d_amdgpu(loss, lossd, u, ud, v, vd, p, pd, q, qd, i_n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_two_field_loss_d_1!(i_n, p, pd, u, ud)
    @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_two_field_loss_d_2!(i_n, q, qd, v, vd)
    for i_seq_x = 1:i_n
        lossd[1] = (lossd[1] + pd[i_seq_x]) + qd[i_seq_x]
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    return nothing
end

function two_field_loss_amdgpu(loss, u, v, p, q, i_n)
    nthread_per_block = 256
    @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_two_field_loss_1!(i_n, p, u)
    @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_two_field_loss_2!(i_n, q, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    return nothing
end
