import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_normcomp_b_1!(i_n, u, v, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = u[i_x] - v[i_x]
    return nothing
end

function cuda_kernel_normcomp_b_2!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_normcomp_b_3!(i_n, lossb, w, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    return nothing
end

function cuda_kernel_normcomp_b_4!(i_n, ub, vb, wb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ub[i_x] = ub[i_x] + wb[i_x]
    vb[i_x] = vb[i_x] + -(wb[i_x])
    wb[i_x] = 0.0
    return nothing
end

function cuda_kernel_normcomp_1!(i_n, u, v, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = u[i_x] - v[i_x]
    return nothing
end

function cuda_kernel_normcomp_2!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x] ^ 2
    return nothing
end

function initstacks_normcomp_b_cuda()
    return nothing
end

function normcomp_b_cuda(loss, lossb, u, ub, v, vb, w, wb, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_normcomp_b_1!(i_n, u, v, w)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_normcomp_b_2!(i_n, loss, w)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_normcomp_b_3!(i_n, lossb, w, wb)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_normcomp_b_4!(i_n, ub, vb, wb)
    return nothing
end

function normcomp_cuda(loss, u, v, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_normcomp_1!(i_n, u, v, w)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_normcomp_2!(i_n, loss, w)
    return nothing
end
