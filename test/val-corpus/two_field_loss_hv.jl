function initstacks_two_field_loss_b()
    return nothing
end

function two_field_loss_hv(loss, lossb, u, ub, v, vb, p, pb, q, qb, i_n, lossd, lossbd, ud, ubd, vd, vbd, pd, pbd, qd, qbd)
    for i_x = 1:i_n
        __hcse_0 = u[i_x]
        pd[i_x] = (2__hcse_0) * ud[i_x]
        p[i_x] = __hcse_0 ^ 2
    end
    for i_x = 1:i_n
        __hcse_1 = v[i_x]
        qd[i_x] = (3 * __hcse_1 ^ 2) * vd[i_x]
        q[i_x] = __hcse_1 ^ 3
    end
    for i_x2 = 1:i_n
        lossd[1] = (lossd[1] + pd[i_x2]) + qd[i_x2]
        loss[1] = loss[1] + p[i_x2] + q[i_x2]
    end
    for i_x2 = i_n:-1:1
        __cse_0d = lossbd[1]
        __cse_0 = lossb[1]
        pbd[i_x2] = pbd[i_x2] + __cse_0d
        pb[i_x2] = pb[i_x2] + __cse_0
        qbd[i_x2] = qbd[i_x2] + __cse_0d
        qb[i_x2] = qb[i_x2] + __cse_0
    end
    for i_x = i_n:-1:1
        __oldb_0d = qbd[i_x]
        __oldb_0 = qb[i_x]
        qbd[i_x] = 0.0
        qb[i_x] = 0.0
        __hcse_2 = v[i_x]
        __hcse_3 = 3 * __hcse_2 ^ 2
        vbd[i_x] = vbd[i_x] + (__oldb_0 * (3 * ((2__hcse_2) * vd[i_x])) + __hcse_3 * __oldb_0d)
        vb[i_x] = vb[i_x] + __hcse_3 * __oldb_0
    end
    for i_x = i_n:-1:1
        __oldb_0d = pbd[i_x]
        __oldb_0 = pb[i_x]
        pbd[i_x] = 0.0
        pb[i_x] = 0.0
        __hcse_4 = 2 * u[i_x]
        ubd[i_x] = ubd[i_x] + (__oldb_0 * (2 * ud[i_x]) + __hcse_4 * __oldb_0d)
        ub[i_x] = ub[i_x] + __hcse_4 * __oldb_0
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
