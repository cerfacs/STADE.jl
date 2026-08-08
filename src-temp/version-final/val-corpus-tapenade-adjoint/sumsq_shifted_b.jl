function initstacks_sumsq_shifted_b()
    return
end
function sumsq_shifted_b(loss, lossb, u, ub, alpha, alphab, beta, betab, i_n)
    for i_seq_x = i_n:-1:1
        tempb = 2 * (alpha * u[i_seq_x] + beta) * lossb[1]
        alphab = alphab + u[i_seq_x] * tempb
        ub[i_seq_x] = ub[i_seq_x] + alpha * tempb
        betab = betab + tempb
    end
    return alphab,betab
end
function sumsq_shifted(loss, u, alpha, beta, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + (alpha * u[i_seq_x] + beta) ^ 2
    end
end
