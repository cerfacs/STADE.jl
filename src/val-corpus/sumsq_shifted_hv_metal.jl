import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function initstacks_sumsq_shifted_b_metal()
    return nothing
end

function sumsq_shifted_hv_metal(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n, lossd, lossbd, ud, ubd, alphad, alphabd, betad, betabd)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * (alpha * u[i_seq_x] + beta)) * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
    end
    for i_seq_x = i_n:-1:1
        alphabd = alphabd + (((2 * (alpha * u[i_seq_x] + beta)) * lossb[1]) * ud[i_seq_x] + u[i_seq_x] * (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1]))
        alphab = alphab + u[i_seq_x] * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
        ubd[i_seq_x] = ubd[i_seq_x] + (((2 * (alpha * u[i_seq_x] + beta)) * lossb[1]) * alphad + alpha * (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1]))
        ub[i_seq_x] = ub[i_seq_x] + alpha * ((2 * (alpha * u[i_seq_x] + beta)) * lossb[1])
        betabd = betabd + (lossb[1] * (2 * ((u[i_seq_x] * alphad + alpha * ud[i_seq_x]) + betad)) + (2 * (alpha * u[i_seq_x] + beta)) * lossbd[1])
        betab = betab + (2 * (alpha * u[i_seq_x] + beta)) * lossb[1]
    end
    return (alphab, alphabd, betab, betabd)
end

function sumsq_shifted_metal(loss, u, alpha, beta, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
    end
    return nothing
end
