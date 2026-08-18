import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_bilinear_hv_1!(a, ad, i_m, i_n, loss, lossd, x, xd, y, yd)
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

function cuda_kernel_bilinear_hv_2!(a, ab, abd, ad, i_m, i_n, lossb, lossbd, x, xb, xbd, xd, y, yb, ybd, yd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_m, -1) + 1
        return nothing
    end
    i_seq_i = i_m + (__tid - 1) * -1
    for i_seq_j = i_n:-1:1
        xbd[i_seq_i] = xbd[i_seq_i] + (lossb[1] * (y[i_seq_j] * ad[i_seq_i, i_seq_j] + a[i_seq_i, i_seq_j] * yd[i_seq_j]) + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossbd[1])
        xb[i_seq_i] = xb[i_seq_i] + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossb[1]
        abd[i_seq_i, i_seq_j] = abd[i_seq_i, i_seq_j] + (lossb[1] * (y[i_seq_j] * xd[i_seq_i] + x[i_seq_i] * yd[i_seq_j]) + (x[i_seq_i] * y[i_seq_j]) * lossbd[1])
        ab[i_seq_i, i_seq_j] = ab[i_seq_i, i_seq_j] + (x[i_seq_i] * y[i_seq_j]) * lossb[1]
        CUDA.@atomic ybd[i_seq_j] += lossb[1] * (a[i_seq_i, i_seq_j] * xd[i_seq_i] + x[i_seq_i] * ad[i_seq_i, i_seq_j]) + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * lossbd[1]
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

function bilinear_hv_cuda(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n, lossd, lossbd, xd, xbd, ad, abd, yd, ybd)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_bilinear_hv_1!(a, ad, i_m, i_n, loss, lossd, x, xd, y, yd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_m, -1) + 1, nthread_per_block) cuda_kernel_bilinear_hv_2!(a, ab, abd, ad, i_m, i_n, lossb, lossbd, x, xb, xbd, xd, y, yb, ybd, yd)
    return nothing
end

function bilinear_cuda(loss, x, a, y, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_bilinear_1!(a, i_m, i_n, loss, x, y)
    return nothing
end
