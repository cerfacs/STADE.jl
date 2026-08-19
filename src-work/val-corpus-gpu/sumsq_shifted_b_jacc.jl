import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_sumsq_shifted_b_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function jacc_kernel_sumsq_shifted_b_2!(__jacc_i, alpha, alphab, beta, betab, i_n, lossb, u, ub)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    Atomix.@atomic alphab[1] += u[i_seq_x] * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    ub[i_seq_x] = ub[i_seq_x] + alpha * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    Atomix.@atomic betab[1] += (2 * (alpha * u[i_seq_x] + beta)) * lossb[1]
    return nothing
end

function jacc_kernel_sumsq_shifted_1!(__jacc_i, loss, __jgen_redval)
    Atomix.@atomic loss[1] += __jgen_redval[1]
    return nothing
end

function initstacks_sumsq_shifted_b_jacc()
    return nothing
end

function sumsq_shifted_b_jacc(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n)
    alphab = JACC.array([alphab])
    betab = JACC.array([betab])
    __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u)->(alpha * u[i_seq_x] + beta) ^ 2))(u))
    JACC.@parallel_for range = 1 jacc_kernel_sumsq_shifted_b_1!(loss, __jgen_redval_1)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_sumsq_shifted_b_2!(alpha, alphab, beta, betab, i_n, lossb, u, ub)
    alphab = (JACC.to_host(alphab))[1]
    betab = (JACC.to_host(betab))[1]
    return (alphab, betab)
end

function sumsq_shifted_jacc(loss, u, alpha, beta, i_n)
    __jgen_redval_1 = JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u)->(alpha * u[i_seq_x] + beta) ^ 2))(u))
    JACC.@parallel_for range = 1 jacc_kernel_sumsq_shifted_1!(loss, __jgen_redval_1)
    return nothing
end
