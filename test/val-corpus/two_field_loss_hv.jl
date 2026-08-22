function initstacks_two_field_loss_b()
    return nothing
end

function two_field_loss_hv(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n, lossd, lossbd, ud, ubd, vd, vbd, pd, pbd, qd, qbd)
    for i_x = 1:i_n
        pd[i_x] = (2 * u[i_x]) * ud[i_x]
        p[i_x] = u[i_x] ^ 2
    end
    for i_x = 1:i_n
        qd[i_x] = (3 * v[i_x] ^ 2) * vd[i_x]
        q[i_x] = v[i_x] ^ 3
    end
    for i_seq_x = 1:i_n
        lossd[1] = (lossd[1] + pd[i_seq_x]) + qd[i_seq_x]
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    for i_seq_x = i_n:-1:1
        pbd[i_seq_x] = pbd[i_seq_x] + lossbd[1]
        pb[i_seq_x] = pb[i_seq_x] + lossb[1]
        qbd[i_seq_x] = qbd[i_seq_x] + lossbd[1]
        qb[i_seq_x] = qb[i_seq_x] + lossb[1]
    end
    for i_x = 1:i_n
        vbd[i_x] = vbd[i_x] + (qb[i_x] * (3 * ((2 * v[i_x]) * vd[i_x])) + (3 * v[i_x] ^ 2) * qbd[i_x])
        vb[i_x] = vb[i_x] + (3 * v[i_x] ^ 2) * qb[i_x]
        qbd[i_x] = 0.0
        qb[i_x] = 0.0
    end
    for i_x = 1:i_n
        ubd[i_x] = ubd[i_x] + (pb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * pbd[i_x])
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * pb[i_x]
        pbd[i_x] = 0.0
        pb[i_x] = 0.0
    end
    return nothing
end

function two_field_loss(loss, u, v, p, q, i_n)
    for i_x = 1:i_n
        p[i_x] = u[i_x] ^ 2
    end
    for i_x = 1:i_n
        q[i_x] = v[i_x] ^ 3
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
end
