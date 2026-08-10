import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function amdgpu_kernel_cond_field_choice_b_1!(i_n, u, w)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function amdgpu_kernel_cond_field_choice_b_2!(i_n, v, w)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function amdgpu_kernel_cond_field_choice_b_3!(i_n, u, ub, wb)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * wb[i_x]
    wb[i_x] = 0.0
    return nothing
end

function amdgpu_kernel_cond_field_choice_b_4!(i_n, v, vb, wb)
    __tid = (workitemIdx()).x + ((workgroupIdx()).x - 1) * (workgroupDim()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vb[i_x] = vb[i_x] + (2 * v[i_x]) * wb[i_x]
    wb[i_x] = 0.0
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

function initstacks_cond_field_choice_b_amdgpu()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function cond_field_choice_b_amdgpu(loss, lossb, u, ub, v, vb, w, wb, i_branch, i_n, branch_stack)
    nthread_per_block = 256
    if i_branch == 1
        push!(branch_stack, 1)
        @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_cond_field_choice_b_1!(i_n, u, w)
    else
        push!(branch_stack, 0)
        @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_cond_field_choice_b_2!(i_n, v, w)
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    end
    __branch = pop!(branch_stack)
    if __branch == 1
        @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_cond_field_choice_b_3!(i_n, u, ub, wb)
    else
        @roc groupsize = nthread_per_block gridsize = cld(div(i_n - 1, 1) + 1, nthread_per_block) amdgpu_kernel_cond_field_choice_b_4!(i_n, v, vb, wb)
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
