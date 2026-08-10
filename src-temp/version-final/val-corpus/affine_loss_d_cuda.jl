import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_affine_loss_d_1!(a, ad, b, bd, i_n, u, ud, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
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

function affine_loss_d_cuda(loss, lossd, u, ud, a, ad, b, bd, v, vd, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_affine_loss_d_1!(a, ad, b, bd, i_n, u, ud, v, vd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * v[i_seq_x]) * vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
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
