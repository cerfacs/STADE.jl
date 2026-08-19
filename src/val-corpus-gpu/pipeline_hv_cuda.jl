import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_pipeline_hv_1!(i_n, u, ud, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vd[i_x] = (2 * u[i_x]) * ud[i_x]
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function cuda_kernel_pipeline_hv_2!(i_n, u, ud, v, vd, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = u[i_x] * vd[i_x] + v[i_x] * ud[i_x]
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function cuda_kernel_pipeline_hv_3!(i_n, loss, lossd, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += wd[i_seq_x]
    CUDA.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function cuda_kernel_pipeline_hv_4!(i_n, lossb, lossbd, wb, wbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    wbd[i_seq_x] = wbd[i_seq_x] + lossbd[1]
    wb[i_seq_x] = wb[i_seq_x] + lossb[1]
    return nothing
end

function cuda_kernel_pipeline_hv_5!(i_n, u, ub, ubd, ud, v, vb, vbd, vd, wb, wbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vbd[i_x] = vbd[i_x] + (wb[i_x] * ud[i_x] + u[i_x] * wbd[i_x])
    vb[i_x] = vb[i_x] + u[i_x] * wb[i_x]
    ubd[i_x] = ubd[i_x] + (wb[i_x] * vd[i_x] + v[i_x] * wbd[i_x])
    ub[i_x] = ub[i_x] + v[i_x] * wb[i_x]
    wbd[i_x] = 0.0
    wb[i_x] = 0.0
    return nothing
end

function cuda_kernel_pipeline_hv_6!(i_n, u, ub, ubd, ud, vb, vbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ubd[i_x] = ubd[i_x] + (vb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * vbd[i_x])
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
    vbd[i_x] = 0.0
    vb[i_x] = 0.0
    return nothing
end

function cuda_kernel_pipeline_1!(i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = u[i_x] ^ 2 + 1.0
    return nothing
end

function cuda_kernel_pipeline_2!(i_n, u, v, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = v[i_x] * u[i_x]
    return nothing
end

function cuda_kernel_pipeline_3!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function initstacks_pipeline_b_cuda()
    return nothing
end

function pipeline_hv_cuda(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_hv_1!(i_n, u, ud, v, vd)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_hv_2!(i_n, u, ud, v, vd, w, wd)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_hv_3!(i_n, loss, lossd, w, wd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_pipeline_hv_4!(i_n, lossb, lossbd, wb, wbd)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_hv_5!(i_n, u, ub, ubd, ud, v, vb, vbd, vd, wb, wbd)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_hv_6!(i_n, u, ub, ubd, ud, vb, vbd)
    return nothing
end

function pipeline_cuda(loss, u, v, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_1!(i_n, u, v)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_2!(i_n, u, v, w)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_3!(i_n, loss, w)
    return nothing
end
