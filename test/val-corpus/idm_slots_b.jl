function initstacks_idm_slots_b()
    return nothing
end

function idm_slots_b(loss, lossb, y, yb, x, xb, w, wb, i_n)
    for i_t = 1:i_n
        i_b = 4 * (i_t - 1)
        __cse_0 = x[i_t]
        __cse_1 = w[1]
        __cse_2 = __cse_0 * __cse_1
        __cse_3 = x[i_t + 1]
        __cse_4 = w[2]
        y[i_b + 1] = __cse_2 + __cse_3 * __cse_4
        __cse_5 = w[3]
        __cse_6 = __cse_3 * __cse_5
        y[i_b + 2] = __cse_0 * __cse_4 + __cse_6
        y[i_b + 3] = __cse_0 * __cse_5 + __cse_3 * __cse_1
        y[i_b + 4] = __cse_2 + __cse_6
    end
    for i_k = 1:4i_n
        __cse_7 = y[i_k]
        loss[1] = loss[1] + __cse_7 * __cse_7
    end
    for i_k = 4i_n:-1:1
        __cse_8 = y[i_k] * lossb[1]
        yb[i_k] = yb[i_k] + __cse_8
        yb[i_k] = yb[i_k] + __cse_8
    end
    for i_t = i_n:-1:1
        i_b = 4 * (i_t - 1)
        __oldb_0 = yb[i_b + 4]
        yb[i_b + 4] = 0.0
        __cse_9 = w[1]
        xb[i_t] = xb[i_t] + __cse_9 * __oldb_0
        __cse_10 = x[i_t]
        wb[1] = wb[1] + __cse_10 * __oldb_0
        __cse_11 = w[3]
        xb[i_t + 1] = xb[i_t + 1] + __cse_11 * __oldb_0
        __cse_12 = x[i_t + 1]
        wb[3] = wb[3] + __cse_12 * __oldb_0
        __oldb_0 = yb[i_b + 3]
        yb[i_b + 3] = 0.0
        xb[i_t] = xb[i_t] + __cse_11 * __oldb_0
        wb[3] = wb[3] + __cse_10 * __oldb_0
        xb[i_t + 1] = xb[i_t + 1] + __cse_9 * __oldb_0
        wb[1] = wb[1] + __cse_12 * __oldb_0
        __oldb_0 = yb[i_b + 2]
        yb[i_b + 2] = 0.0
        __cse_13 = w[2]
        xb[i_t] = xb[i_t] + __cse_13 * __oldb_0
        wb[2] = wb[2] + __cse_10 * __oldb_0
        xb[i_t + 1] = xb[i_t + 1] + __cse_11 * __oldb_0
        wb[3] = wb[3] + __cse_12 * __oldb_0
        __oldb_0 = yb[i_b + 1]
        yb[i_b + 1] = 0.0
        xb[i_t] = xb[i_t] + __cse_9 * __oldb_0
        wb[1] = wb[1] + __cse_10 * __oldb_0
        xb[i_t + 1] = xb[i_t + 1] + __cse_13 * __oldb_0
        wb[2] = wb[2] + __cse_12 * __oldb_0
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
