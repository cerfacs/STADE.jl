import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_geomrecur_d_1!(i_n, loss, lossd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * u[i_seq_x]) * ud[i_seq_x]
    CUDA.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_geomrecur_1!(i_n, loss, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += u[i_seq_x] ^ 2
    return nothing
end

function geomrecur_d_cuda(loss, lossd, u, ud, c, cd, i_n)
    nthread_per_block = 256
    for i_seq_x = 2:i_n
        CUDA.@allowscalar begin
                ud[i_seq_x] = u[i_seq_x - 1] * cd + c * ud[i_seq_x - 1]
                u[i_seq_x] = c * u[i_seq_x - 1]
            end
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_geomrecur_d_1!(i_n, loss, lossd, u, ud)
    return nothing
end

function geomrecur_cuda(loss, u, c, i_n)
    nthread_per_block = 256
    for i_seq_x = 2:i_n
        CUDA.@allowscalar begin
                u[i_seq_x] = c * u[i_seq_x - 1]
            end
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_geomrecur_1!(i_n, loss, u)
    return nothing
end
