import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_matvec_loss_d_1!(a, ad, i_m, i_n, u, ud, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        vd[i_i] = vd[i_i] + (u[i_seq_j] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * ud[i_seq_j])
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function cuda_kernel_matvec_loss_d_2!(i_m, loss, lossd, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_seq_i = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * v[i_seq_i]) * vd[i_seq_i]
    CUDA.@atomic loss[1] += v[i_seq_i] ^ 2
    return nothing
end

function cuda_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function cuda_kernel_matvec_loss_2!(i_m, loss, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_seq_i = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += v[i_seq_i] ^ 2
    return nothing
end

function matvec_loss_d_cuda(loss, lossd, a, ad, u, ud, v, vd, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_matvec_loss_d_1!(a, ad, i_m, i_n, u, ud, v, vd)
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_matvec_loss_d_2!(i_m, loss, lossd, v, vd)
    return nothing
end

function matvec_loss_cuda(loss, a, u, v, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_matvec_loss_2!(i_m, loss, v)
    return nothing
end
