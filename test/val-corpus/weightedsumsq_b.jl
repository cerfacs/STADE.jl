function initstacks_weightedsumsq_b()
    return nothing
end

function weightedsumsq_b(loss, lossb, u, ub, w, wb, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + w[i_x] * u[i_x] ^ 2
    end
    for i_x = i_n:-1:1
        wb[i_x] = wb[i_x] + u[i_x] ^ 2 * lossb[1]
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * (w[i_x] * lossb[1])
    end
    return nothing
end

function weightedsumsq(loss, u, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + w[i_x] * u[i_x] ^ 2
    end
end
