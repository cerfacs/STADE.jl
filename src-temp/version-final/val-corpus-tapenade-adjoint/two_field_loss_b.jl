function initstacks_two_field_loss_b()
    return
end
function two_field_loss_b(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n)
    for i_seq_x = i_n:-1:1
        pb[i_seq_x] = pb[i_seq_x] + lossb[1]
        qb[i_seq_x] = qb[i_seq_x] + lossb[1]
    end
    for i_x = 1:i_n
        vb[i_x] = vb[i_x] + 3 * v[i_x] ^ 2 * qb[i_x]
        qb[i_x] = 0.0
    end
    for i_x = 1:i_n
        ub[i_x] = ub[i_x] + 2 * u[i_x] * pb[i_x]
        pb[i_x] = 0.0
    end
    return 
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
