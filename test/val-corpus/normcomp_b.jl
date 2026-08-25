function initstacks_normcomp_b()
    return nothing
end

function normcomp_b(loss, lossb, u, ub, v, vb, w, wb, i_n)
    for i_x = 1:i_n
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + w[i_x2] ^ 2
    end
    for i_x2 = i_n:-1:1
        wb[i_x2] = wb[i_x2] + (2 * w[i_x2]) * lossb[1]
    end
    for i_x = i_n:-1:1
        ub[i_x] = ub[i_x] + wb[i_x]
        vb[i_x] = vb[i_x] + -(wb[i_x])
        wb[i_x] = 0.0
    end
    return nothing
end

function normcomp(loss, u, v, w, i_n)
    for i_x = 1:i_n
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + w[i_x2] ^ 2
    end
end
