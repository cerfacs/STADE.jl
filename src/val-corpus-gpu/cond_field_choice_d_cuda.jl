import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_cond_field_choice_d_1!(i_n, u, ud, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = (2 * u[i_x]) * ud[i_x]
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function cuda_kernel_cond_field_choice_d_2!(i_n, v, vd, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    wd[i_x] = (2 * v[i_x]) * vd[i_x]
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function cuda_kernel_cond_field_choice_d_3!(i_n, loss, lossd, w, wd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += wd[i_seq_x]
    CUDA.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function cuda_kernel_cond_field_choice_1!(i_n, u, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = u[i_x] ^ 2
    return nothing
end

function cuda_kernel_cond_field_choice_2!(i_n, v, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    w[i_x] = v[i_x] ^ 2
    return nothing
end

function cuda_kernel_cond_field_choice_3!(i_n, loss, w)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += w[i_seq_x]
    return nothing
end

function cond_field_choice_d_cuda(loss, lossd, u, ud, v, vd, w, wd, i_branch, i_n)
    nthread_per_block = 256
    if i_branch == 1
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_d_1!(i_n, u, ud, w, wd)
    else
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_d_2!(i_n, v, vd, w, wd)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_d_3!(i_n, loss, lossd, w, wd)
    return nothing
end

function cond_field_choice_cuda(loss, u, v, w, i_branch, i_n)
    nthread_per_block = 256
    if i_branch == 1
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_1!(i_n, u, w)
    else
        @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_2!(i_n, v, w)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_cond_field_choice_3!(i_n, loss, w)
    return nothing
end
