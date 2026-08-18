import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_clamped_sumsq_b_1!(branch_stack, i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    if u[i_seq_x] > 0.0
        branch_stack[(i_seq_x - 1) + 1] = 1
        w = u[i_seq_x] ^ 2
    else
        branch_stack[(i_seq_x - 1) + 1] = 0
        w = 0.0
    end
    CUDA.@atomic loss[1] += w
    return nothing
end

function cuda_kernel_clamped_sumsq_b_2!(branch_stack, i_n, lossb, u, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    wb = 0.0
    wb = wb + lossb[1]
    __branch = branch_stack[(i_seq_x - 1) + 1]
    if __branch == 1
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * wb
        wb = 0.0
    else
        wb = 0.0
    end
    return nothing
end

function cuda_kernel_clamped_sumsq_1!(i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    if u[i_seq_x] > 0.0
        w = u[i_seq_x] ^ 2
    else
        w = 0.0
    end
    CUDA.@atomic loss[1] += w
    return nothing
end

function initstacks_clamped_sumsq_b_cuda(i_n)
    branch_stack = CuArray{Int64}(undef, div(i_n - 1, 1) + 1)
    return branch_stack
end

function clamped_sumsq_b_cuda(loss, lossb, u, ub, i_n, branch_stack)
    nthread_per_block = 256
    w = 0.0
    wb = 0.0
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_clamped_sumsq_b_1!(branch_stack, i_n, loss, u)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_clamped_sumsq_b_2!(branch_stack, i_n, lossb, u, ub)
    return nothing
end

function clamped_sumsq_cuda(loss, u, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_clamped_sumsq_1!(i_n, loss, u)
    return nothing
end
