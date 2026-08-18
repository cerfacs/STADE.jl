import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_sumsq_shifted_hv_1!(alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * (alpha * u[i_seq_x] + beta)) * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)
    CUDA.@atomic loss[1] += (alpha * u[i_seq_x] + beta) ^ 2
    return nothing
end

function cuda_kernel_sumsq_shifted_hv_2!(alpha, alphab, alphabd, alphad, beta, betab, betabd, betad, i_n, lossb, lossbd, u, ub, ubd, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    CUDA.@atomic alphabd[1] += ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1]) * ud[i_seq_x] + u[i_seq_x] * (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1])
    CUDA.@atomic alphab[1] += u[i_seq_x] * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    ubd[i_seq_x] = ubd[i_seq_x] + (((2 * (alpha * u[i_seq_x] + beta)) * lossb[1]) * alphad + alpha * (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1]))
    ub[i_seq_x] = ub[i_seq_x] + alpha * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    CUDA.@atomic betabd[1] += lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1]
    CUDA.@atomic betab[1] += (2 * (alpha * u[i_seq_x] + beta)) * lossb[1]
    return nothing
end

function cuda_kernel_sumsq_shifted_1!(alpha, beta, i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += (alpha * u[i_seq_x] + beta) ^ 2
    return nothing
end

function initstacks_sumsq_shifted_b_cuda()
    return nothing
end

function sumsq_shifted_hv_cuda(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n, lossd, lossbd, ud, ubd, alphad, alphabd, betad, betabd)
    alphab = CuArray([alphab])
    alphabd = CuArray([alphabd])
    betab = CuArray([betab])
    betabd = CuArray([betabd])
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_hv_1!(alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_hv_2!(alpha, alphab, alphabd, alphad, beta, betab, betabd, betad, i_n, lossb, lossbd, u, ub, ubd, ud)
    alphab = (Array(alphab))[1]
    alphabd = (Array(alphabd))[1]
    betab = (Array(betab))[1]
    betabd = (Array(betabd))[1]
    return (alphab, alphabd, betab, betabd)
end

function sumsq_shifted_cuda(loss, u, alpha, beta, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_1!(alpha, beta, i_n, loss, u)
    return nothing
end
