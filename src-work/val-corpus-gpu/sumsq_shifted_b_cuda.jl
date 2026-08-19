import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_sumsq_shifted_b_1!(alpha, alphab, beta, betab, i_n, lossb, u, ub)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    CUDA.@atomic alphab[1] += u[i_seq_x] * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    ub[i_seq_x] = ub[i_seq_x] + alpha * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    CUDA.@atomic betab[1] += (2 * (alpha * u[i_seq_x] + beta)) * lossb[1]
    return nothing
end

function initstacks_sumsq_shifted_b_cuda()
    return nothing
end

function sumsq_shifted_b_cuda(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n)
    alphab = CuArray([alphab])
    betab = CuArray([betab])
    nthread_per_block = 256
    CUDA.@allowscalar begin
            loss[1] = loss[1] + mapreduce(((__mr_1,)->(alpha * __mr_1 + beta) ^ 2), +, u)
        end
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_sumsq_shifted_b_1!(alpha, alphab, beta, betab, i_n, lossb, u, ub)
    alphab = (Array(alphab))[1]
    betab = (Array(betab))[1]
    return (alphab, betab)
end

function sumsq_shifted_cuda(loss, u, alpha, beta, i_n)
    CUDA.@allowscalar begin
            loss[1] = loss[1] + mapreduce(((__mr_1,)->(alpha * __mr_1 + beta) ^ 2), +, u)
        end
    return nothing
end
