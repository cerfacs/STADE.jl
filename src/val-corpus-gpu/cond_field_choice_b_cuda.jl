import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_cond_field_choice_b_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function cuda_kernel_cond_field_choice_b_2!(i_n, v, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function cuda_kernel_cond_field_choice_b_3!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function cuda_kernel_cond_field_choice_b_4!(i_n, lossb, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    return nothing
end

function cuda_kernel_cond_field_choice_b_5!(i_n, u, ub, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * wb[i_x]
    wb[i_x] = 0.0
    return nothing
end

function cuda_kernel_cond_field_choice_b_6!(i_n, v, vb, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vb[i_x] = vb[i_x] + (2 * v[i_x]) * wb[i_x]
    wb[i_x] = 0.0
    return nothing
end

function cuda_kernel_cond_field_choice_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function cuda_kernel_cond_field_choice_2!(i_n, v, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function cuda_kernel_cond_field_choice_3!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function initstacks_cond_field_choice_b_cuda()
    branch_stack = CuArray{Int64}(undef, 1)
    return branch_stack
end

function cond_field_choice_b_cuda(loss, lossb, u, ub, v, vb, w, wb, i_branch, i_n, branch_stack)
    nthread_per_block = 256
    if i_branch == 1
        CUDA.@allowscalar begin
                branch_stack[1] = 1
            end
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_b_1!(i_n, u, w)
    else
        CUDA.@allowscalar begin
                branch_stack[1] = 0
            end
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_b_2!(i_n, v, w)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_b_3!(i_n, loss, w)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_b_4!(i_n, lossb, wb)
    CUDA.@allowscalar begin
            __branch = branch_stack[1]
        end
    if __branch == 1
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_b_5!(i_n, u, ub, wb)
    else
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_b_6!(i_n, v, vb, wb)
    end
    return nothing
end

function cond_field_choice_cuda(loss, u, v, w, i_branch, i_n)
    nthread_per_block = 256
    if i_branch == 1
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_1!(i_n, u, w)
    else
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_2!(i_n, v, w)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_3!(i_n, loss, w)
    return nothing
end
