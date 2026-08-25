function initstacks_sumsq_shifted_b()
    return nothing
end

function sumsq_shifted_b(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_x] + beta) ^ 2
    end
    for i_x = i_n:-1:1
        alphab = alphab + u[i_x] * ((2 * (alpha * u[i_x] + beta)) * lossb[1])
        ub[i_x] = ub[i_x] + alpha * ((2 * (alpha * u[i_x] + beta)) * lossb[1])
        betab = betab + (2 * (alpha * u[i_x] + beta)) * lossb[1]
    end
    return (alphab, betab)
end

function sumsq_shifted(loss, u, alpha, beta, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_x] + beta) ^ 2
    end
end
