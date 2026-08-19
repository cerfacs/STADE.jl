import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_cond_loop_choice_d_1!(i_n, loss, lossd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * u[i_seq_x]) * ud[i_seq_x]
    CUDA.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_cond_loop_choice_d_2!(i_n, loss, lossd, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * v[i_seq_x]) * vd[i_seq_x]
    CUDA.@atomic loss[1] += v[i_seq_x] ^ 2
    return nothing
end

function cond_loop_choice_d_cuda(loss, lossd, u, ud, v, vd, i_branch, i_n)
    nthread_per_block = 256
    if i_branch == 1
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_loop_choice_d_1!(i_n, loss, lossd, u, ud)
    else
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_loop_choice_d_2!(i_n, loss, lossd, v, vd)
    end
    return nothing
end

function cond_loop_choice_cuda(loss, u, v, i_branch, i_n)
    if i_branch == 1
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(abs2, u)
            end
    else
        CUDA.@allowscalar begin
                loss[1] = loss[1] + sum(abs2, v)
            end
    end
    return nothing
end
