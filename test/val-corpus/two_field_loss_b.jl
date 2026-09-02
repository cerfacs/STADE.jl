function initstacks_two_field_loss_b()
    return nothing
end

function two_field_loss_b(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n)
    for i_x = 1:i_n
        p[i_x] = u[i_x] ^ 2
    end
    for i_x = 1:i_n
        q[i_x] = v[i_x] ^ 3
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + p[i_x2] + q[i_x2]
    end
    for i_x2 = i_n:-1:1
        __cse_0 = lossb[1]
        pb[i_x2] = pb[i_x2] + __cse_0
        qb[i_x2] = qb[i_x2] + __cse_0
    end
    for i_x = i_n:-1:1
        __oldb_0 = qb[i_x]
        qb[i_x] = 0.0
        vb[i_x] = vb[i_x] + (3 * v[i_x] ^ 2) * __oldb_0
    end
    for i_x = i_n:-1:1
        __oldb_0 = pb[i_x]
        pb[i_x] = 0.0
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * __oldb_0
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
    for i_x2 = 1:i_n
        loss[1] = loss[1] + p[i_x2] + q[i_x2]
    end
end
