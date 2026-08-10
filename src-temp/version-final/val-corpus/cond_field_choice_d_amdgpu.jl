import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_cond_field_choice_d_1!(i_n, u, ud, w, wd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = (2 * u[i_x]) * ud[i_x]
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function amdgpu_kernel_cond_field_choice_d_2!(i_n, v, vd, w, wd)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = (2 * v[i_x]) * vd[i_x]
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function amdgpu_kernel_cond_field_choice_1!(i_n, u, w)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function amdgpu_kernel_cond_field_choice_2!(i_n, v, w)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function cond_field_choice_d_amdgpu(loss, lossd, u, ud, v, vd, w, wd, i_branch, i_n)
    nthread_per_block = 256
    if i_branch == 1
        @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_cond_field_choice_d_1!(i_n, u, ud, w, wd)
    else
        @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_cond_field_choice_d_2!(i_n, v, vd, w, wd)
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end

function cond_field_choice_amdgpu(loss, u, v, w, i_branch, i_n)
    nthread_per_block = 256
    if i_branch == 1
        @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_cond_field_choice_1!(i_n, u, w)
    else
        @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_cond_field_choice_2!(i_n, v, w)
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end
