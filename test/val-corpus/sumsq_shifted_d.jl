function sumsq_shifted_d(loss, lossd, u, ud, alpha, alphad, beta, betad, i_n)
    for i_x = 1:i_n
        __cse_0 = u[i_x]
        __cse_1 = alpha * __cse_0 + beta
        lossd[1] = lossd[1] + (2__cse_1) * ((__cse_0 * alphad + alpha * ud[i_x]) + betad)
        loss[1] = loss[1] + __cse_1 ^ 2
    end
    return nothing
end

function sumsq_shifted(loss, u, alpha, beta, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_x] + beta) ^ 2
    end
end
