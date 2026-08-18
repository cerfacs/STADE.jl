import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_sumsq_shifted_d_1!(__jacc_i, alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * (alpha * u[i_seq_x] + beta)) * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)
    Atomix.@atomic loss[1] += (alpha * u[i_seq_x] + beta) ^ 2
    return nothing
end

function jacc_kernel_sumsq_shifted_1!(__jacc_i, alpha, beta, i_n, loss, u)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic loss[1] += (alpha * u[i_seq_x] + beta) ^ 2
    return nothing
end

function sumsq_shifted_d_jacc(loss, lossd, u, ud, alpha, alphad, beta, betad, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_sumsq_shifted_d_1!(alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    return nothing
end

function sumsq_shifted_jacc(loss, u, alpha, beta, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_sumsq_shifted_1!(alpha, beta, i_n, loss, u)
    return nothing
end
