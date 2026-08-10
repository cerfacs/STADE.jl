import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_normcomp_hv_1!(i_n, u, ud, v, vd, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = ud[i_x] + -(vd[i_x])
    w[i_x] = u[i_x] - v[i_x]
    return nothing
end

function cuda_kernel_normcomp_hv_2!(i_n, ub, ubd, vb, vbd, wb, wbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ubd[i_x] = ubd[i_x] + wbd[i_x]
    ub[i_x] = ub[i_x] + wb[i_x]
    vbd[i_x] = vbd[i_x] + -(wbd[i_x])
    vb[i_x] = vb[i_x] + -(wb[i_x])
    wbd[i_x] = 0.0
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

function initstacks_normcomp_b_cuda()
    return nothing
end

function normcomp_hv_cuda(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_normcomp_hv_1!(i_n, u, ud, v, vd, w, wd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * (2 * wd[i_seq_x]) + (2 * w[i_seq_x]) * lossbd[1])
        wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_normcomp_hv_2!(i_n, ub, ubd, vb, vbd, wb, wbd)
    return nothing
end

function normcomp_cuda(loss, u, v, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_normcomp_1!(i_n, u, v, w)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end
