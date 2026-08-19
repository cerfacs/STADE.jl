import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_sumsq_shifted_b_1!(alpha, beta, i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += (alpha * u[i_seq_x] + beta) ^ 2
    return nothing
end

function cuda_kernel_sumsq_shifted_b_2!(alpha, alphab, beta, betab, i_n, lossb, u, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    CUDA.@atomic alphab[1] += u[i_seq_x] * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    ub[i_seq_x] = ub[i_seq_x] + alpha * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    CUDA.@atomic betab[1] += (2 * (alpha * u[i_seq_x] + beta)) * lossb[1]
    return nothing
end

function cuda_kernel_sumsq_shifted_1!(alpha, beta, i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += (alpha * u[i_seq_x] + beta) ^ 2
    return nothing
end

function initstacks_sumsq_shifted_b_cuda()
    return nothing
end

function sumsq_shifted_b_cuda(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n)
    alphab = CuArray([alphab])
    betab = CuArray([betab])
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_b_1!(alpha, beta, i_n, loss, u)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_b_2!(alpha, alphab, beta, betab, i_n, lossb, u, ub)
    alphab = (Array(alphab))[1]
    betab = (Array(betab))[1]
    return (alphab, betab)
end

function sumsq_shifted_cuda(loss, u, alpha, beta, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_1!(alpha, beta, i_n, loss, u)
    return nothing
end
