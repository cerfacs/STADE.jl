import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add("JACC")
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

function jacc_kernel_two_field_loss_b_3!(__jacc_i, i_n, qb, v, vb)
    i_x = 1 + (__jacc_i - 1)
    vb[i_x] = vb[i_x] + (3 * v[i_x] ^ 2) * qb[i_x]
    qb[i_x] = 0.0
    return nothing
end

function jacc_kernel_two_field_loss_b_4!(__jacc_i, i_n, pb, u, ub)
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

function initstacks_two_field_loss_b_jacc()
    return nothing
end

function two_field_loss_b_jacc(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_two_field_loss_b_1!, i_n, p, u)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_two_field_loss_b_2!, i_n, q, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        pb[i_seq_x] = pb[i_seq_x] + lossb[1]
        qb[i_seq_x] = qb[i_seq_x] + lossb[1]
    end
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_two_field_loss_b_3!, i_n, qb, v, vb)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_two_field_loss_b_4!, i_n, pb, u, ub)
    return nothing
end

function two_field_loss_jacc(loss, u, v, p, q, i_n)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_two_field_loss_1!, i_n, p, u)
    JACC.parallel_for(div(i_n - 1, 1) + 1, jacc_kernel_two_field_loss_2!, i_n, q, v)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    return nothing
end
