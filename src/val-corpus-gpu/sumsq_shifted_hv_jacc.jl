import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_sumsq_shifted_hv_1!(__jacc_i, alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    i_seq_x = 1 + (__jacc_i - 1)
    Atomix.@atomic lossd[1] += (2 * (alpha * u[i_seq_x] + beta)) * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)
    Atomix.@atomic loss[1] += (alpha * u[i_seq_x] + beta) ^ 2
    return nothing
end

function jacc_kernel_sumsq_shifted_hv_2!(__jacc_i, alpha, alphab, alphabd, alphad, beta, betab, betabd, betad, i_n, lossb, lossbd, u, ub, ubd, ud)
    i_seq_x = i_n + (__jacc_i - 1) * -1
    Atomix.@atomic alphabd[1] += ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1]) * ud[i_seq_x] + u[i_seq_x] * (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1])
    Atomix.@atomic alphab[1] += u[i_seq_x] * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    ubd[i_seq_x] = ubd[i_seq_x] + (((2 * (alpha * u[i_seq_x] + beta)) * lossb[1]) * alphad + alpha * (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1]))
    ub[i_seq_x] = ub[i_seq_x] + alpha * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
    Atomix.@atomic betabd[1] += lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1]
    Atomix.@atomic betab[1] += (2 * (alpha * u[i_seq_x] + beta)) * lossb[1]
    return nothing
end

function initstacks_sumsq_shifted_b_jacc()
    return nothing
end

function sumsq_shifted_hv_jacc(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n, lossd, lossbd, ud, ubd, alphad, alphabd, betad, betabd)
    alphab = JACC.array([alphab])
    alphabd = JACC.array([alphabd])
    betab = JACC.array([betab])
    betabd = JACC.array([betabd])
    JACC.@parallel_for range = div(i_n - 1, 1) + 1 jacc_kernel_sumsq_shifted_hv_1!(alpha, alphad, beta, betad, i_n, loss, lossd, u, ud)
    JACC.@parallel_for range = div(1 - i_n, -1) + 1 jacc_kernel_sumsq_shifted_hv_2!(alpha, alphab, alphabd, alphad, beta, betab, betabd, betad, i_n, lossb, lossbd, u, ub, ubd, ud)
    alphab = (JACC.to_host(alphab))[1]
    alphabd = (JACC.to_host(alphabd))[1]
    betab = (JACC.to_host(betab))[1]
    betabd = (JACC.to_host(betabd))[1]
    return (alphab, alphabd, betab, betabd)
end

function sumsq_shifted_jacc(loss, u, alpha, beta, i_n)
    loss[1] = loss[1] + JACC.@parallel_reduce(range = div(i_n - 1, 1) + 1, (((i_seq_x, u)->(alpha * u[i_seq_x] + beta) ^ 2))(u))
    return nothing
end
