import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_bilinear_b_1!(a, i_m, i_n, loss, x, y)
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

function cuda_kernel_bilinear_b_2!(a, ab, i_m, i_n, lossb, x, xb, y, yb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_m, -1) + 1
        return nothing
    end
    i_seq_i = i_m + (__tid - 1) * -1
    for i_seq_j = i_n:-1:1
        xb[i_seq_i] = xb[i_seq_i] + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossb[1]
        ab[i_seq_i, i_seq_j] = ab[i_seq_i, i_seq_j] + (x[i_seq_i] * y[i_seq_j]) * lossb[1]
        CUDA.@atomic yb[i_seq_j] += (x[i_seq_i] * a[i_seq_i, i_seq_j]) * lossb[1]
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

function initstacks_bilinear_b_cuda()
    return nothing
end

function bilinear_b_cuda(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_bilinear_b_1!(a, i_m, i_n, loss, x, y)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_m, -1) + 1, nthread_per_block) cuda_kernel_bilinear_b_2!(a, ab, i_m, i_n, lossb, x, xb, y, yb)
    return nothing
end

function bilinear_cuda(loss, x, a, y, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_bilinear_1!(a, i_m, i_n, loss, x, y)
    return nothing
end
