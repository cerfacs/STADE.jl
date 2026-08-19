import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_weightedsumsq_d_1!(i_n, loss, lossd, u, ud, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += u[i_seq_x] ^ 2 * wd[i_seq_x] + w[i_seq_x] * ((2 * u[i_seq_x]) * ud[i_seq_x])
    CUDA.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
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

function weightedsumsq_d_cuda(loss, lossd, u, ud, w, wd, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_weightedsumsq_d_1!(i_n, loss, lossd, u, ud, w, wd)
    return nothing
end

function weightedsumsq_cuda(loss, u, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_weightedsumsq_1!(i_n, loss, u, w)
    return nothing
end
