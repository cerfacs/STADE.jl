import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_relu_field_b_1!(branch_stack, i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    if u[i_x] > 0.0
        branch_stack[(i_x - 1) + 1] = 1
        v[i_x] = u[i_x] ^ 2
    else
        branch_stack[(i_x - 1) + 1] = 0
        v[i_x] = 0.0
    end
    return nothing
end

function cuda_kernel_relu_field_b_2!(i_n, loss, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += v[i_seq_x]
    return nothing
end

function cuda_kernel_relu_field_b_3!(i_n, lossb, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    vb[i_seq_x] = vb[i_seq_x] + lossb[1]
    return nothing
end

function cuda_kernel_relu_field_b_4!(branch_stack, i_n, u, ub, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_x = i_n + (__tid - 1) * -1
    __branch = branch_stack[(i_x - 1) + 1]
    if __branch == 1
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
        vb[i_x] = 0.0
    else
        vb[i_x] = 0.0
    end
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

function initstacks_relu_field_b_cuda(i_n)
    branch_stack = CuArray{Int64}(undef, div(i_n - 1, 1) + 1)
    return branch_stack
end

function relu_field_b_cuda(loss, lossb, u, ub, v, vb, i_n, branch_stack)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_relu_field_b_1!(branch_stack, i_n, u, v)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_relu_field_b_2!(i_n, loss, v)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_relu_field_b_3!(i_n, lossb, vb)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_relu_field_b_4!(branch_stack, i_n, u, ub, vb)
    return nothing
end

function relu_field_cuda(loss, u, v, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_relu_field_1!(i_n, u, v)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_relu_field_2!(i_n, loss, v)
    return nothing
end
