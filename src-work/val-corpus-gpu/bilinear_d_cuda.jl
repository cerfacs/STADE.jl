import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_bilinear_d_1!(a, ad, i_m, i_n, loss, lossd, x, xd, y, yd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_seq_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        CUDA.@atomic lossd[1] += (a[i_seq_i, i_seq_j] * y[i_seq_j]) * xd[i_seq_i] + (x[i_seq_i] * y[i_seq_j]) * ad[i_seq_i, i_seq_j] + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * yd[i_seq_j]
        CUDA.@atomic loss[1] += x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
    end
    return nothing
end

function cuda_kernel_bilinear_1!(a, i_m, i_n, loss, x, y)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_seq_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        CUDA.@atomic loss[1] += x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
    end
    return nothing
end

function bilinear_d_cuda(loss, lossd, x, xd, a, ad, y, yd, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_bilinear_d_1!(a, ad, i_m, i_n, loss, lossd, x, xd, y, yd)
    return nothing
end

function bilinear_cuda(loss, x, a, y, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_bilinear_1!(a, i_m, i_n, loss, x, y)
    return nothing
end
