import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_two_field_loss_hv_1!(i_n, p, pd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    pd[i_x] = (2 * u[i_x]) * ud[i_x]
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function cuda_kernel_two_field_loss_hv_2!(i_n, q, qd, v, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    qd[i_x] = (3 * v[i_x] ^ 2) * vd[i_x]
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function cuda_kernel_two_field_loss_hv_3!(i_n, qb, qbd, v, vb, vbd, vd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vbd[i_x] = vbd[i_x] + (qb[i_x] * (3 * ((2 * v[i_x]) * vd[i_x])) + (3 * v[i_x] ^ 2) * qbd[i_x])
    vb[i_x] = vb[i_x] + (3 * v[i_x] ^ 2) * qb[i_x]
    qbd[i_x] = 0.0
    qb[i_x] = 0.0
    return nothing
end

function cuda_kernel_two_field_loss_hv_4!(i_n, pb, pbd, u, ub, ubd, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ubd[i_x] = ubd[i_x] + (pb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * pbd[i_x])
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * pb[i_x]
    pbd[i_x] = 0.0
    pb[i_x] = 0.0
    return nothing
end

function cuda_kernel_two_field_loss_1!(i_n, p, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function cuda_kernel_two_field_loss_2!(i_n, q, v)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function initstacks_two_field_loss_b_cuda()
    return nothing
end

function two_field_loss_hv_cuda(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n, lossd, lossbd, ud, ubd, vd, vbd, pd, pbd, qd, qbd)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_two_field_loss_hv_1!(i_n, p, pd, u, ud)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_two_field_loss_hv_2!(i_n, q, qd, v, vd)
    for i_seq_x = 1:i_n
        lossd[1] = (lossd[1] + pd[i_seq_x]) + qd[i_seq_x]
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        pbd[i_seq_x] = pbd[i_seq_x] + lossbd[1]
        pb[i_seq_x] = pb[i_seq_x] + lossb[1]
        qbd[i_seq_x] = qbd[i_seq_x] + lossbd[1]
        qb[i_seq_x] = qb[i_seq_x] + lossb[1]
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_two_field_loss_hv_3!(i_n, qb, qbd, v, vb, vbd, vd)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_two_field_loss_hv_4!(i_n, pb, pbd, u, ub, ubd, ud)
    return nothing
end

function two_field_loss_cuda(loss, u, v, p, q, i_n)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_two_field_loss_1!(i_n, p, u)
    @cuda threads = nthread_per_block blocks = cld(div(i_n - 1, 1) + 1, nthread_per_block) cuda_kernel_two_field_loss_2!(i_n, q, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    return nothing
end
