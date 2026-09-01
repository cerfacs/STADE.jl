function two_field_loss_d(loss, lossd, u, ud, v, vd, p, pd, q, qd, i_n)
    for i_x = 1:i_n
        __cse_0 = u[i_x]
        pd[i_x] = (2__cse_0) * ud[i_x]
        p[i_x] = __cse_0 ^ 2
    end
    for i_x = 1:i_n
        __cse_1 = v[i_x]
        qd[i_x] = (3 * __cse_1 ^ 2) * vd[i_x]
        q[i_x] = __cse_1 ^ 3
    end
    for i_x2 = 1:i_n
        lossd[1] = (lossd[1] + pd[i_x2]) + qd[i_x2]
        loss[1] = loss[1] + p[i_x2] + q[i_x2]
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
