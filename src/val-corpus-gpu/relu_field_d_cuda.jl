import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_relu_field_d_1!(i_n, u, ud, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    if u[i_x] > 0.0
        vd[i_x] = (2 * u[i_x]) * ud[i_x]
        v[i_x] = u[i_x] ^ 2
    else
        vd[i_x] = 0.0
        v[i_x] = 0.0
    end
    return nothing
end

function cuda_kernel_relu_field_d_2!(i_n, loss, lossd, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += vd[i_seq_x]
    CUDA.@atomic loss[1] += v[i_seq_x]
    return nothing
end

function cuda_kernel_relu_field_1!(i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    if u[i_x] > 0.0
        v[i_x] = u[i_x] ^ 2
    else
        v[i_x] = 0.0
    end
    return nothing
end

function cuda_kernel_relu_field_2!(i_n, loss, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += v[i_seq_x]
    return nothing
end

function relu_field_d_cuda(loss, lossd, u, ud, v, vd, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_relu_field_d_1!(i_n, u, ud, v, vd)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_relu_field_d_2!(i_n, loss, lossd, v, vd)
    return nothing
end

function relu_field_cuda(loss, u, v, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_relu_field_1!(i_n, u, v)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_relu_field_2!(i_n, loss, v)
    return nothing
end
