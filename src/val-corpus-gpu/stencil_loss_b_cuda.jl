import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_stencil_loss_b_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    w[i_x] = (u[i_x - 1] - 2.0 * u[i_x]) + u[i_x + 1]
    return nothing
end

function cuda_kernel_stencil_loss_b_2!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_seq_x = 2 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_stencil_loss_b_3!(i_n, lossb, w, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(2 - (i_n - 1), -1) + 1
        return nothing
    end
    i_seq_x = (i_n - 1) + (__tid - 1) * -1
    wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    return nothing
end

function cuda_kernel_stencil_loss_b_4!(i_n, ub, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div((i_n - 1) - 2, 1) + 1
        return nothing
    end
    i_x = 2 + (__tid - 1)
    ub[i_x - 1] = ub[i_x - 1] + wb[i_x]
    ub[i_x] = ub[i_x] + 2.0 * -(wb[i_x])
    ub[i_x + 1] = ub[i_x + 1] + wb[i_x]
    wb[i_x] = 0.0
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

function initstacks_stencil_loss_b_cuda()
    return nothing
end

function stencil_loss_b_cuda(loss, lossb, u, ub, w, wb, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_b_1!(i_n, u, w)
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_b_2!(i_n, loss, w)
    @cuda threads = nthread_per_block blocks = cld(div(2 - (i_n - 1), -1) + 1, nthread_per_block) cuda_kernel_stencil_loss_b_3!(i_n, lossb, w, wb)
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_b_4!(i_n, ub, wb)
    return nothing
end

function stencil_loss_cuda(loss, u, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_1!(i_n, u, w)
    @cuda threads = nthread_per_block blocks = cld(div((i_n - 1) - 2, 1) + 1, nthread_per_block) cuda_kernel_stencil_loss_2!(i_n, loss, w)
    return nothing
end
