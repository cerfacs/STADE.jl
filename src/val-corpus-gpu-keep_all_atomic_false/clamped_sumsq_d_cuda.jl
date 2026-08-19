import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_clamped_sumsq_d_1!(i_n, loss, lossd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    if u[i_seq_x] > 0.0
        wd = (2 * u[i_seq_x]) * ud[i_seq_x]
        w = u[i_seq_x] ^ 2
    else
        wd = 0.0
        w = 0.0
    end
    CUDA.@atomic lossd[1] += wd
    CUDA.@atomic loss[1] += w
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

function clamped_sumsq_d_cuda(loss, lossd, u, ud, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_clamped_sumsq_d_1!(i_n, loss, lossd, u, ud)
    return nothing
end

function clamped_sumsq_cuda(loss, u, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_clamped_sumsq_1!(i_n, loss, u)
    return nothing
end
