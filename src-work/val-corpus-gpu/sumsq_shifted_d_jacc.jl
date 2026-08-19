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

function jacc_kernel_sumsq_shifted_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function sumsq_shifted_d_jacc(loss, lossd, u, ud, alpha, alphad, beta, betad, i_n)
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_sumsq_shifted_d_1!(alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    return nothing
end

function sumsq_shifted_jacc(loss, u, alpha, beta, i_n)
    __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u)->(alpha * u[i_seq_x] + beta) ^ 2))(u))
    JACC.@parallel_for range = 1 jacc_kernel_sumsq_shifted_1!(loss, __jgen_redval_1)
    return nothing
end
