function idm_slots_d(loss, lossd, y, yd, x, xd, w, wd, i_n)
    for i_t = 1:i_n
        i_bd = 0.0
        i_b = 4 * (i_t - 1)
        __cse_0 = w[1]
        __cse_1 = xd[i_t]
        __cse_2 = x[i_t]
        __cse_3 = wd[1]
        __cse_4 = __cse_0 * __cse_1 + __cse_2 * __cse_3
        __cse_5 = w[2]
        __cse_6 = xd[i_t + 1]
        __cse_7 = x[i_t + 1]
        __cse_8 = wd[2]
        yd[i_b + 1] = __cse_4 + (__cse_5 * __cse_6 + __cse_7 * __cse_8)
        __cse_9 = __cse_2 * __cse_0
        y[i_b + 1] = __cse_9 + __cse_7 * __cse_5
        __cse_10 = w[3]
        __cse_11 = wd[3]
        __cse_12 = __cse_10 * __cse_6 + __cse_7 * __cse_11
        yd[i_b + 2] = (__cse_5 * __cse_1 + __cse_2 * __cse_8) + __cse_12
        __cse_13 = __cse_7 * __cse_10
        y[i_b + 2] = __cse_2 * __cse_5 + __cse_13
        yd[i_b + 3] = (__cse_10 * __cse_1 + __cse_2 * __cse_11) + (__cse_0 * __cse_6 + __cse_7 * __cse_3)
        y[i_b + 3] = __cse_2 * __cse_10 + __cse_7 * __cse_0
        yd[i_b + 4] = __cse_4 + __cse_12
        y[i_b + 4] = __cse_9 + __cse_13
    end
    for i_k = 1:4i_n
        __cse_14 = y[i_k]
        __cse_15 = __cse_14 * yd[i_k]
        lossd[1] = lossd[1] + (__cse_15 + __cse_15)
        loss[1] = loss[1] + __cse_14 * __cse_14
    end
    return nothing
end

function idm_slots(loss, y, x, w, i_n)
    for i_t = 1:i_n
        i_b = 4 * (i_t - 1)
        y[i_b + 1] = x[i_t] * w[1] + x[i_t + 1] * w[2]
        y[i_b + 2] = x[i_t] * w[2] + x[i_t + 1] * w[3]
        y[i_b + 3] = x[i_t] * w[3] + x[i_t + 1] * w[1]
        y[i_b + 4] = x[i_t] * w[1] + x[i_t + 1] * w[3]
    end
    for i_k = 1:4i_n
        loss[1] = loss[1] + y[i_k] * y[i_k]
    end
end
