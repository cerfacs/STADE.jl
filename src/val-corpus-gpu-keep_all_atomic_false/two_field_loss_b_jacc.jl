import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_two_field_loss_b_1!(__jacc_i, i_n, p, u)
    i_x = 1 + (__jacc_i - 1)
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function jacc_kernel_two_field_loss_b_2!(__jacc_i, i_n, q, v)
    i_x = 1 + (__jacc_i - 1)
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function jacc_kernel_two_field_loss_b_3!(__jacc_i, i_n, loss, p, q)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += p[i_seq_x] + q[i_seq_x]
    return nothing
end

function jacc_kernel_two_field_loss_b_4!(__jacc_i, i_n, lossb, pb, qb)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    pb[i_seq_x] = pb[i_seq_x] + lossb[1]
    qb[i_seq_x] = qb[i_seq_x] + lossb[1]
    return nothing
end

function jacc_kernel_two_field_loss_b_5!(__jacc_i, i_n, qb, v, vb)
    i_x = 1 + (__jacc_i - 1)
    vb[i_x] = vb[i_x] + (3 * v[i_x] ^ 2) * qb[i_x]
    qb[i_x] = 0.0
    return nothing
end

function jacc_kernel_two_field_loss_b_6!(__jacc_i, i_n, pb, u, ub)
    i_x = 1 + (__jacc_i - 1)
    ub[i_x] = ub[i_x] + (2 * u[i_x]) * pb[i_x]
    pb[i_x] = 0.0
    return nothing
end

function jacc_kernel_two_field_loss_1!(__jacc_i, i_n, p, u)
    i_x = 1 + (__jacc_i - 1)
    p[i_x] = u[i_x] ^ 2
    return nothing
end

function jacc_kernel_two_field_loss_2!(__jacc_i, i_n, q, v)
    i_x = 1 + (__jacc_i - 1)
    q[i_x] = v[i_x] ^ 3
    return nothing
end

function jacc_kernel_two_field_loss_3!(__jacc_i, i_n, loss, p, q)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += p[i_seq_x] + q[i_seq_x]
    return nothing
end

function initstacks_two_field_loss_b_jacc()
    return nothing
end

function two_field_loss_b_jacc(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_two_field_loss_b_1!(i_n, p, u)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_two_field_loss_b_2!(i_n, q, v)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_two_field_loss_b_3!(i_n, loss, p, q)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_two_field_loss_b_4!(i_n, lossb, pb, qb)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_two_field_loss_b_5!(i_n, qb, v, vb)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_two_field_loss_b_6!(i_n, pb, u, ub)
    return nothing
end

function two_field_loss_jacc(loss, u, v, p, q, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_two_field_loss_1!(i_n, p, u)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_two_field_loss_2!(i_n, q, v)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_two_field_loss_3!(i_n, loss, p, q)
    return nothing
end
