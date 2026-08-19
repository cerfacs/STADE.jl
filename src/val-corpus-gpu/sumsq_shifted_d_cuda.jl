import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_sumsq_shifted_d_1!(alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * (alpha * u[i_seq_x] + beta)) * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)
    CUDA.@atomic loss[1] += (alpha * u[i_seq_x] + beta) ^ 2
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

function sumsq_shifted_d_cuda(loss, lossd, u, ud, alpha, alphad, beta, betad, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_d_1!(alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    return nothing
end

function sumsq_shifted_cuda(loss, u, alpha, beta, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_1!(alpha, beta, i_n, loss, u)
    return nothing
end
