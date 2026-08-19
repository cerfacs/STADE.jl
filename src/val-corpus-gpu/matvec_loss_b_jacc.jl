import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_matvec_loss_b_1!(__jacc_i, a, i_m, i_n, u, v)
    i_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function jacc_kernel_matvec_loss_b_2!(__jacc_i, i_m, lossb, v, vb)
    i_seq_i = i_m + (__jacc_i - 1) * -1
    vb[i_seq_i] = vb[i_seq_i] + (2 * v[i_seq_i]) * lossb[1]
    return nothing
end

function jacc_kernel_matvec_loss_b_3!(__jacc_i, a, ab, i_m, i_n, u, ub, vb)
    i_i = 1 + (__jacc_i - 1)
    for i_seq_j = i_n:-1:1
        ab[i_i, i_seq_j] = ab[i_i, i_seq_j] + u[i_seq_j] * vb[i_i]
        Atomix.@atomic ub[i_seq_j] += a[i_i, i_seq_j] * vb[i_i]
    end
    return nothing
end

function jacc_kernel_matvec_loss_1!(__jacc_i, a, i_m, i_n, u, v)
    i_i = 1 + (__jacc_i - 1)
    for i_seq_j = 1:i_n
        v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
    end
    return nothing
end

function initstacks_matvec_loss_b_jacc()
    return nothing
end

function matvec_loss_b_jacc(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_b_1!(a, i_m, i_n, u, v)
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_m - 1, 1) + 1, (((i_seq_i, v)->v[i_seq_i] ^ 2))(v))
    JACC.@parallel_for range = div(1 - i_m, -1) + 1 jacc_kernel_matvec_loss_b_2!(i_m, lossb, v, vb)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_b_3!(a, ab, i_m, i_n, u, ub, vb)
    return nothing
end

function matvec_loss_jacc(loss, a, u, v, i_m, i_n)
    JACC.@parallel_for range = div(i_m - 1, 1) + 1 jacc_kernel_matvec_loss_1!(a, i_m, i_n, u, v)
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_m - 1, 1) + 1, (((i_seq_i, v)->v[i_seq_i] ^ 2))(v))
    return nothing
end
