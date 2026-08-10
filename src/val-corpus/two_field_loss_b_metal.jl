import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function metal_kernel_two_field_loss_b_1!(i_n, p, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function metal_kernel_two_field_loss_b_2!(i_n, q, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function metal_kernel_two_field_loss_b_3!(i_n, qb, v, vb)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    vb[i_x] = vb[i_x] + (3 * v[i_x] ^ 2) * qb[i_x]
    qb[i_x] = 0.0f0
    return nothing
end

function metal_kernel_two_field_loss_b_4!(i_n, pb, u, ub)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * pb[i_x]
    pb[i_x] = 0.0f0
    return nothing
end

function metal_kernel_two_field_loss_1!(i_n, p, u)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function metal_kernel_two_field_loss_2!(i_n, q, v)
    __tid = (thread_position_in_grid()).x
    if __tid > div(i_n - 1, 1) + 1
        return nothing
    end
    i_x = 1 + (__tid - 1)
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function initstacks_two_field_loss_b_metal()
    return nothing
end

function two_field_loss_b_metal(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_two_field_loss_b_1!(i_n, p, u)
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_two_field_loss_b_2!(i_n, q, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        pb[i_seq_x] = pb[i_seq_x] + lossb[1]
        qb[i_seq_x] = qb[i_seq_x] + lossb[1]
    end
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_two_field_loss_b_3!(i_n, qb, v, vb)
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_two_field_loss_b_4!(i_n, pb, u, ub)
    return nothing
end

function two_field_loss_metal(loss, u, v, p, q, i_n)
    nthread_per_block = 256
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_two_field_loss_1!(i_n, p, u)
    @metal threads = nthread_per_block groups = cld(div(i_n - 1, 1) + 1, nthread_per_block) metal_kernel_two_field_loss_2!(i_n, q, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    return nothing
end
