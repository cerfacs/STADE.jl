import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_weightedsumsq_b_1!(i_n, loss, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_weightedsumsq_b_2!(i_n, lossb, u, ub, w, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    wb[i_seq_x] = wb[i_seq_x] + u[i_seq_x] ^ 2 * lossb[1]
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * (w[i_seq_x] * lossb[1])
    return nothing
end

function cuda_kernel_weightedsumsq_1!(i_n, loss, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function initstacks_weightedsumsq_b_cuda()
    return nothing
end

function weightedsumsq_b_cuda(loss, lossb, u, ub, w, wb, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_weightedsumsq_b_1!(i_n, loss, u, w)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_weightedsumsq_b_2!(i_n, lossb, u, ub, w, wb)
    return nothing
end

function weightedsumsq_cuda(loss, u, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_weightedsumsq_1!(i_n, loss, u, w)
    return nothing
end
