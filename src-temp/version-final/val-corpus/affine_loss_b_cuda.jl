import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_affine_loss_b_1!(a, b, i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function cuda_kernel_affine_loss_b_2!(a, ab, bb, i_n, u, ub, vb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
    ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
    bb[i_x] = bb[i_x] + vb[i_x]
    vb[i_x] = 0.0
    return nothing
end

function cuda_kernel_affine_loss_1!(a, b, i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function initstacks_affine_loss_b_cuda()
    return nothing
end

function affine_loss_b_cuda(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_affine_loss_b_1!(a, b, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_affine_loss_b_2!(a, ab, bb, i_n, u, ub, vb)
    return nothing
end

function affine_loss_cuda(loss, u, a, b, v, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_affine_loss_1!(a, b, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    return nothing
end
