import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_affine_loss_b_1!(a, b, i_n, u, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function metal_kernel_affine_loss_b_2!(a, ab, bb, i_n, u, ub, vb)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
    ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
    bb[i_x] = bb[i_x] + vb[i_x]
    vb[i_x] = 0.0f0
    return nothing
end

function metal_kernel_affine_loss_1!(a, b, i_n, u, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    v[i_x] = a[i_x] * u[i_x] + b[i_x]
    return nothing
end

function initstacks_affine_loss_b_metal()
    return nothing
end

function affine_loss_b_metal(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_affine_loss_b_1!(a, b, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    end
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_affine_loss_b_2!(a, ab, bb, i_n, u, ub, vb)
    return nothing
end

function affine_loss_metal(loss, u, a, b, v, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_affine_loss_1!(a, b, i_n, u, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    return nothing
end
