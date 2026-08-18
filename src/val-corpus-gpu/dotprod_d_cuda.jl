import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_dotprod_d_1!(i_n, loss, lossd, u, ud, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += v[i_seq_x] * ud[i_seq_x] + u[i_seq_x] * vd[i_seq_x]
    CUDA.@atomic loss[1] += u[i_seq_x] * v[i_seq_x]
    return nothing
end

function cuda_kernel_dotprod_1!(i_n, loss, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += u[i_seq_x] * v[i_seq_x]
    return nothing
end

function dotprod_d_cuda(loss, lossd, u, ud, v, vd, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_dotprod_d_1!(i_n, loss, lossd, u, ud, v, vd)
    return nothing
end

function dotprod_cuda(loss, u, v, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_dotprod_1!(i_n, loss, u, v)
    return nothing
end
