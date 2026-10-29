function while_two_seq_d(loss, lossd, u, ud, a, ad, v, vd, w, wd, i_n)
    i_kd = 0.0
    i_k = 1
    while i_k <= i_n
        __cse_0 = u[i_k]
        __cse_1 = a[i_k]
        vd[i_k] = __cse_0 * ad[i_k] + __cse_1 * ud[i_k]
        v[i_k] = __cse_1 * __cse_0
        i_kd = 0.0
        i_k = i_k + 1
    end
    i_qd = 0.0
    i_q = 1
    while i_q < i_n
        __cse_2 = v[i_q + 1]
        __cse_3 = v[i_q]
        wd[i_q] = __cse_2 * vd[i_q] + __cse_3 * vd[i_q + 1]
        w[i_q] = __cse_3 * __cse_2
        i_qd = 0.0
        i_q = i_q + 1
    end
    for i_x = 1:i_n
        __cse_4 = v[i_x]
        __cse_5 = w[i_x]
        lossd[1] = (lossd[1] + (2__cse_4) * vd[i_x]) + (2__cse_5) * wd[i_x]
        loss[1] = loss[1] + __cse_4 ^ 2 + __cse_5 ^ 2
    end
    return nothing
end

function while_two_seq(loss, u, a, v, w, i_n)
    i_k = 1
    while i_k <= i_n
        v[i_k] = a[i_k] * u[i_k]
        i_k = i_k + 1
    end
    i_q = 1
    while i_q < i_n
        w[i_q] = v[i_q] * v[i_q + 1]
        i_q = i_q + 1
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + v[i_x] ^ 2 + w[i_x] ^ 2
    end
end
