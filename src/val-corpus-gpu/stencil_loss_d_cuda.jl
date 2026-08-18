import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_stencil_loss_d_1!(i_n, u, ud, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    wd[i_x] = (ud[i_x - 1] + -(2.0 * ud[i_x])) + ud[i_x + 1]
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function cuda_kernel_stencil_loss_d_2!(i_n, loss, lossd, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_seq_x = 2 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * w[i_seq_x]) * wd[i_seq_x]
    CUDA.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_stencil_loss_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function cuda_kernel_stencil_loss_2!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_seq_x = 2 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function stencil_loss_d_cuda(loss, lossd, u, ud, w, wd, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_d_1!(i_n, u, ud, w, wd)
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_d_2!(i_n, loss, lossd, w, wd)
    return nothing
end

function stencil_loss_cuda(loss, u, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_1!(i_n, u, w)
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_2!(i_n, loss, w)
    return nothing
end
