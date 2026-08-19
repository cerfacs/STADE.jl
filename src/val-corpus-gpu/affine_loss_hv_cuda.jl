import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_affine_loss_hv_1!(a, ad, b, bd, i_n, u, ud, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function cuda_kernel_affine_loss_hv_2!(i_n, loss, lossd, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_seq_x = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * v[i_seq_x]) * vd[i_seq_x]
    CUDA.@atomic loss[1] += v[i_seq_x] ^ 2
    return nothing
end

function cuda_kernel_affine_loss_hv_3!(i_n, lossb, lossbd, v, vb, vbd, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_n, -1) + 1
        return nothing
    end
    i_seq_x = i_n + (__tid - 1) * -1
    vbd[i_seq_x] = vbd[i_seq_x] + (lossb[1] * (2 * vd[i_seq_x]) + (2 * v[i_seq_x]) * lossbd[1])
    vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    return nothing
end

function cuda_kernel_affine_loss_hv_4!(a, ab, abd, ad, bb, bbd, i_n, u, ub, ubd, ud, vb, vbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    abd[i_x] = abd[i_x] + (vb[i_x] * ud[i_x] + u[i_x] * vbd[i_x])
    ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
    ubd[i_x] = ubd[i_x] + (vb[i_x] * ad[i_x] + a[i_x] * vbd[i_x])
    ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
    bbd[i_x] = bbd[i_x] + vbd[i_x]
    bb[i_x] = bb[i_x] + vb[i_x]
    vbd[i_x] = 0.0
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

function affine_loss_hv_cuda(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, bd, bbd, vd, vbd)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_affine_loss_hv_1!(a, ad, b, bd, i_n, u, ud, v, vd)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_affine_loss_hv_2!(i_n, loss, lossd, v, vd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_n, -1) + 1, nthread_per_block) cuda_kernel_affine_loss_hv_3!(i_n, lossb, lossbd, v, vb, vbd, vd)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_affine_loss_hv_4!(a, ab, abd, ad, bb, bbd, i_n, u, ub, ubd, ud, vb, vbd)
    return nothing
end

function affine_loss_cuda(loss, u, a, b, v, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_affine_loss_1!(a, b, i_n, u, v)
    CUDA.@allowscalar begin
            loss[1] = loss[1] + sum(abs2, v)
        end
    return nothing
end
