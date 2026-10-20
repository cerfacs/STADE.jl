function red_two_locals_d(loss, lossd, u, ud, v, vd, i_n)
    pd = 0.0
    p = 0.0
    for i_x = 1:i_n
        __cse_0 = v[i_x]
        __cse_1 = u[i_x]
        pd = pd + (__cse_0 * ud[i_x] + __cse_1 * vd[i_x])
        p = p + __cse_1 * __cse_0
    end
    qd = 0.0
    q = 0.0
    for i_y = 1:i_n
        __cse_2 = v[i_y]
        __cse_3 = __cse_2 * vd[i_y]
        qd = qd + (__cse_3 + __cse_3)
        q = q + __cse_2 * __cse_2
    end
    lossd[1] = lossd[1] + (q * pd + p * qd)
    loss[1] = loss[1] + p * q
    return nothing
end

function red_two_locals(loss, u, v, i_n)
    p = 0.0
    for i_x = 1:i_n
        p = p + u[i_x] * v[i_x]
    end
    q = 0.0
    for i_y = 1:i_n
        q = q + v[i_y] * v[i_y]
    end
    loss[1] = loss[1] + p * q
end
