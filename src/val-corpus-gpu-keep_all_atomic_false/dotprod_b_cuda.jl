import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_dotprod_b_1!(i_n, lossb, u, ub, v, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    ub[i_seq_x] = ub[i_seq_x] + v[i_seq_x] * lossb[1]
    vb[i_seq_x] = vb[i_seq_x] + u[i_seq_x] * lossb[1]
    return nothing
end

function initstacks_dotprod_b_cuda()
    return nothing
end

function dotprod_b_cuda(loss, lossb, u, ub, v, vb, i_n)
    nthread_per_block = 256
    CUDA.@allowscalar begin
            loss[1] = loss[1] + dot(u, v)
        end
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_dotprod_b_1!(i_n, lossb, u, ub, v, vb)
    return nothing
end

function dotprod_cuda(loss, u, v, i_n)
    CUDA.@allowscalar begin
            loss[1] = loss[1] + dot(u, v)
        end
    return nothing
end
