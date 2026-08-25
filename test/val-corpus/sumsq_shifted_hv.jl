function initstacks_sumsq_shifted_b()
    return nothing
end

function sumsq_shifted_hv(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n, lossd, lossbd, ud, ubd, alphad, alphabd, betad, betabd)
    for i_x = 1:i_n
        lossd[1] = lossd[1] + (2 * (alpha * u[i_x] + beta)) * ((u[i_x] * alphad + alpha * ud[i_x]) + betad)
        loss[1] = loss[1] + (alpha * u[i_x] + beta) ^ 2
    end
    for i_x = i_n:-1:1
        alphabd = alphabd + (((2 * (alpha * u[i_x] + beta)) * lossb[1]) * ud[i_x] + u[i_x] * (lossb[1] * (2 * ((u[i_x] * alphad + alpha * ud[i_x]) + betad)) + (2 * (alpha * u[i_x] + beta)) * lossbd[1]))
        alphab = alphab + u[i_x] * ((2 * (alpha * u[i_x] + beta)) * lossb[1])
        ubd[i_x] = ubd[i_x] + (((2 * (alpha * u[i_x] + beta)) * lossb[1]) * alphad + alpha * (lossb[1] * (2 * ((u[i_x] * alphad + alpha * ud[i_x]) + betad)) + (2 * (alpha * u[i_x] + beta)) * lossbd[1]))
        ub[i_x] = ub[i_x] + alpha * ((2 * (alpha * u[i_x] + beta)) * lossb[1])
        betabd = betabd + (lossb[1] * (2 * ((u[i_x] * alphad + alpha * ud[i_x]) + betad)) + (2 * (alpha * u[i_x] + beta)) * lossbd[1])
        betab = betab + (2 * (alpha * u[i_x] + beta)) * lossb[1]
    end
    return (alphab, alphabd, betab, betabd)
end

function sumsq_shifted(loss, u, alpha, beta, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_x] + beta) ^ 2
    end
end
