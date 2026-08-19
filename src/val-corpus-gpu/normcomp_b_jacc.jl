import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_normcomp_b_1!(__jacc_i, i_n, u, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = u[i_x] - v[i_x]
    return nothing
end

function jacc_kernel_normcomp_b_2!(__jacc_i, i_n, lossb, w, wb)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    return nothing
end

function jacc_kernel_normcomp_b_3!(__jacc_i, i_n, ub, vb, wb)
    i_x = 1 + (__jacc_i - 1)
    ub[i_x] = ub[i_x] + wb[i_x]
    vb[i_x] = vb[i_x] + -(wb[i_x])
    wb[i_x] = 0.0
    return nothing
end

function jacc_kernel_normcomp_1!(__jacc_i, i_n, u, v, w)
    i_x = 1 + (__jacc_i - 1)
    w[i_x] = u[i_x] - v[i_x]
    return nothing
end

function initstacks_normcomp_b_jacc()
    return nothing
end

function normcomp_b_jacc(loss, lossb, u, ub, v, vb, w, wb, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_normcomp_b_1!(i_n, u, v, w)
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, w)->w[i_seq_x] ^ 2))(w))
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_normcomp_b_2!(i_n, lossb, w, wb)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_normcomp_b_3!(i_n, ub, vb, wb)
    return nothing
end

function normcomp_jacc(loss, u, v, w, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_normcomp_1!(i_n, u, v, w)
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, w)->w[i_seq_x] ^ 2))(w))
    return nothing
end
