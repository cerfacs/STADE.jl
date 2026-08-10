import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

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

function pipeline_cuda(loss, u, v, w, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_1!(i_n, u, v)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_pipeline_2!(i_n, u, v, w)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end
