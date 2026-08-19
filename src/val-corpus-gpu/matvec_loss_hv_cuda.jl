import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_matvec_loss_hv_1!(a, ad, i_m, i_n, u, ud, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        vd[i_i] = vd[i_i] + (u[i_seq_j] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * ud[i_seq_j])
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function cuda_kernel_matvec_loss_hv_2!(i_m, loss, lossd, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_seq_i = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * v[i_seq_i]) * vd[i_seq_i]
    CUDA.@atomic loss[1] += v[i_seq_i] ^ 2
    return nothing
end

function cuda_kernel_matvec_loss_hv_3!(i_m, lossb, lossbd, v, vb, vbd, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_m, -1) + 1
        return nothing
    end
    i_seq_i = i_m + (__tid - 1) * -1
    vbd[i_seq_i] = vbd[i_seq_i] + (lossb[1] * (2 * vd[i_seq_i]) + (2 * v[i_seq_i]) * lossbd[1])
    vb[i_seq_i] = vb[i_seq_i] + (2 * v[i_seq_i]) * lossb[1]
    return nothing
end

function cuda_kernel_matvec_loss_hv_4!(a, ab, abd, ad, i_m, i_n, u, ub, ubd, ud, vb, vbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i = 1 + (__tid - 1)
    for i_seq_j = i_n:-1:1
        abd[i_i, i_seq_j] = abd[i_i, i_seq_j] + (vb[i_i] * ud[i_seq_j] + u[i_seq_j] * vbd[i_i])
        ab[i_i, i_seq_j] = ab[i_i, i_seq_j] + u[i_seq_j] * vb[i_i]
        CUDA.@atomic ubd[i_seq_j] += vb[i_i] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * vbd[i_i]
        CUDA.@atomic ub[i_seq_j] += a[i_i, i_seq_j] * vb[i_i]
    end
    return nothing
end

function cuda_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_m - 1, 1) + 1
        return nothing
    end
    i_i = 1 + (__tid - 1)
    for i_seq_j = 1:i_n
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function initstacks_matvec_loss_b_cuda()
    return nothing
end

function matvec_loss_hv_cuda(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n, lossd, lossbd, ad, abd, ud, ubd, vd, vbd)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_matvec_loss_hv_1!(a, ad, i_m, i_n, u, ud, v, vd)
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_matvec_loss_hv_2!(i_m, loss, lossd, v, vd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_m, -1) + 1, nthread_per_block) cuda_kernel_matvec_loss_hv_3!(i_m, lossb, lossbd, v, vb, vbd, vd)
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_matvec_loss_hv_4!(a, ab, abd, ad, i_m, i_n, u, ub, ubd, ud, vb, vbd)
    return nothing
end

function matvec_loss_cuda(loss, a, u, v, i_m, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_m - 1, 1) + 1, nthread_per_block) cuda_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    CUDA.@allowscalar begin
            loss[1] = loss[1] + sum(abs2, v)
        end
    return nothing
end
