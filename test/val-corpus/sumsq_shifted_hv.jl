function initstacks_sumsq_shifted_b()
    return nothing
end

function sumsq_shifted_hv(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n, lossd, lossbd, ud, ubd, alphad, alphabd, betad, betabd)
    for i_x = 1:i_n
        __cse_2 = u[i_x]
        __cse_3 = alpha * __cse_2 + beta
        lossd[1] = lossd[1] + (2__cse_3) * ((__cse_2 * alphad + alpha * ud[i_x]) + betad)
        loss[1] = loss[1] + __cse_3 ^ 2
    end
    for i_x = i_n:-1:1
        __cse_0d = ud[i_x]
        __cse_0 = u[i_x]
        __cse_4 = lossb[1]
        __cse_5 = 2 * (alpha * __cse_0 + beta)
        __cse_1d = __cse_4 * (2 * ((__cse_0 * alphad + alpha * __cse_0d) + betad)) + __cse_5 * lossbd[1]
        __cse_1 = __cse_5 * __cse_4
        alphabd = alphabd + (__cse_1 * __cse_0d + __cse_0 * __cse_1d)
        alphab = alphab + __cse_0 * __cse_1
        ubd[i_x] = ubd[i_x] + (__cse_1 * alphad + alpha * __cse_1d)
        ub[i_x] = ub[i_x] + alpha * __cse_1
        betabd = betabd + __cse_1d
        betab = betab + __cse_1
    end
    return (alphab, alphabd, betab, betabd)
end

function sumsq_shifted(loss, u, alpha, beta, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_x] + beta) ^ 2
    end
end
