import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_weightedsumsq_b_1!(__jacc_i, i_n, loss, u, w)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function jacc_kernel_weightedsumsq_b_2!(__jacc_i, i_n, lossb, u, ub, w, wb)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    wb[i_seq_x] = wb[i_seq_x] + u[i_seq_x] ^ 2 * lossb[1]
    ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * (w[i_seq_x] * lossb[1])
    return nothing
end

function jacc_kernel_weightedsumsq_1!(__jacc_i, i_n, loss, u, w)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += w[i_seq_x] * u[i_seq_x] ^ 2
    return nothing
end

function initstacks_weightedsumsq_b_jacc()
    return nothing
end

function weightedsumsq_b_jacc(loss, lossb, u, ub, w, wb, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_weightedsumsq_b_1!(i_n, loss, u, w)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_weightedsumsq_b_2!(i_n, lossb, u, ub, w, wb)
    return nothing
end

function weightedsumsq_jacc(loss, u, w, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_weightedsumsq_1!(i_n, loss, u, w)
    return nothing
end
