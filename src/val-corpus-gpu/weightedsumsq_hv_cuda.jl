import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_weightedsumsq_hv_1!(i_n, loss, lossd, u, ud, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += u[i_seq_x] ^ 2 * wd[i_seq_x] + w[i_seq_x] * ((2 * u[i_seq_x]) * ud[i_seq_x])
    CUDA.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_weightedsumsq_hv_2!(i_n, lossb, lossbd, u, ub, ubd, ud, w, wb, wbd, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    wbd[i_seq_x] = wbd[i_seq_x] + (lossb[1] * ((2 * u[i_seq_x]) * ud[i_seq_x]) + u[i_seq_x] ^ 2 * lossbd[1])
    wb[i_seq_x] = wb[i_seq_x] + u[i_seq_x] ^ 2 * lossb[1]
    ubd[i_seq_x] = ubd[i_seq_x] + ((w[i_seq_x] * lossb[1]) * (2 * ud[i_seq_x]) + (2 * u[i_seq_x]) * (lossb[1] * wd[i_seq_x] + w[i_seq_x] * lossbd[1]))
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * (w[i_seq_x] * lossb[1])
    return nothing
end

function initstacks_weightedsumsq_b_cuda()
    return nothing
end

function weightedsumsq_hv_cuda(loss, lossb, u, ub, w, wb, i_n, lossd, lossbd, ud, ubd, wd, wbd)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_weightedsumsq_hv_1!(i_n, loss, lossd, u, ud, w, wd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_weightedsumsq_hv_2!(i_n, lossb, lossbd, u, ub, ubd, ud, w, wb, wbd, wd)
    return nothing
end

function weightedsumsq_cuda(loss, u, w, i_n)
    CUDA.@allowscalar begin
            loss[1] = loss[1] + mapreduce(((__mr_1, __mr_2)->__mr_2 * __mr_1 ^ 2), +, u, w)
        end
    return nothing
end
