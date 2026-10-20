function initstacks_idm_slots_b()
    return nothing
end

function idm_slots_hv(loss, lossb, y, yb, x, xb, w, wb, i_n, lossd, lossbd, yd, ybd, xd, xbd, wd, wbd)
    for i_t = 1:i_n
        i_b = 4 * (i_t - 1)
        __cse_0d = xd[i_t]
        __cse_0 = x[i_t]
        __cse_1d = wd[1]
        __cse_1 = w[1]
        __cse_2d = __cse_1 * __cse_0d + __cse_0 * __cse_1d
        __cse_2 = __cse_0 * __cse_1
        __cse_3d = xd[i_t + 1]
        __cse_3 = x[i_t + 1]
        __cse_4d = wd[2]
        __cse_4 = w[2]
        yd[i_b + 1] = __cse_2d + (__cse_4 * __cse_3d + __cse_3 * __cse_4d)
        y[i_b + 1] = __cse_2 + __cse_3 * __cse_4
        __cse_5d = wd[3]
        __cse_5 = w[3]
        __cse_6d = __cse_5 * __cse_3d + __cse_3 * __cse_5d
        __cse_6 = __cse_3 * __cse_5
        yd[i_b + 2] = (__cse_4 * __cse_0d + __cse_0 * __cse_4d) + __cse_6d
        y[i_b + 2] = __cse_0 * __cse_4 + __cse_6
        yd[i_b + 3] = (__cse_5 * __cse_0d + __cse_0 * __cse_5d) + (__cse_1 * __cse_3d + __cse_3 * __cse_1d)
        y[i_b + 3] = __cse_0 * __cse_5 + __cse_3 * __cse_1
        yd[i_b + 4] = __cse_2d + __cse_6d
        y[i_b + 4] = __cse_2 + __cse_6
    end
    for i_k = 1:4i_n
        __cse_7d = yd[i_k]
        __cse_7 = y[i_k]
        __hcse_0 = __cse_7 * __cse_7d
        lossd[1] = lossd[1] + (__hcse_0 + __hcse_0)
        loss[1] = loss[1] + __cse_7 * __cse_7
    end
    for i_k = 4i_n:-1:1
        __hcse_1 = lossb[1]
        __hcse_2 = y[i_k]
        __cse_8d = __hcse_1 * yd[i_k] + __hcse_2 * lossbd[1]
        __cse_8 = __hcse_2 * __hcse_1
        ybd[i_k] = ybd[i_k] + __cse_8d
        yb[i_k] = yb[i_k] + __cse_8
        ybd[i_k] = ybd[i_k] + __cse_8d
        yb[i_k] = yb[i_k] + __cse_8
    end
    for i_t = i_n:-1:1
        i_b = 4 * (i_t - 1)
        __oldb_0d = ybd[i_b + 4]
        __oldb_0 = yb[i_b + 4]
        ybd[i_b + 4] = 0.0
        yb[i_b + 4] = 0.0
        __cse_9d = wd[1]
        __cse_9 = w[1]
        xbd[i_t] = xbd[i_t] + (__oldb_0 * __cse_9d + __cse_9 * __oldb_0d)
        xb[i_t] = xb[i_t] + __cse_9 * __oldb_0
        __cse_10d = xd[i_t]
        __cse_10 = x[i_t]
        wbd[1] = wbd[1] + (__oldb_0 * __cse_10d + __cse_10 * __oldb_0d)
        wb[1] = wb[1] + __cse_10 * __oldb_0
        __cse_11d = wd[3]
        __cse_11 = w[3]
        xbd[i_t + 1] = xbd[i_t + 1] + (__oldb_0 * __cse_11d + __cse_11 * __oldb_0d)
        xb[i_t + 1] = xb[i_t + 1] + __cse_11 * __oldb_0
        __cse_12d = xd[i_t + 1]
        __cse_12 = x[i_t + 1]
        wbd[3] = wbd[3] + (__oldb_0 * __cse_12d + __cse_12 * __oldb_0d)
        wb[3] = wb[3] + __cse_12 * __oldb_0
        __oldb_0d = ybd[i_b + 3]
        __oldb_0 = yb[i_b + 3]
        ybd[i_b + 3] = 0.0
        yb[i_b + 3] = 0.0
        xbd[i_t] = xbd[i_t] + (__oldb_0 * __cse_11d + __cse_11 * __oldb_0d)
        xb[i_t] = xb[i_t] + __cse_11 * __oldb_0
        wbd[3] = wbd[3] + (__oldb_0 * __cse_10d + __cse_10 * __oldb_0d)
        wb[3] = wb[3] + __cse_10 * __oldb_0
        xbd[i_t + 1] = xbd[i_t + 1] + (__oldb_0 * __cse_9d + __cse_9 * __oldb_0d)
        xb[i_t + 1] = xb[i_t + 1] + __cse_9 * __oldb_0
        wbd[1] = wbd[1] + (__oldb_0 * __cse_12d + __cse_12 * __oldb_0d)
        wb[1] = wb[1] + __cse_12 * __oldb_0
        __oldb_0d = ybd[i_b + 2]
        __oldb_0 = yb[i_b + 2]
        ybd[i_b + 2] = 0.0
        yb[i_b + 2] = 0.0
        __cse_13d = wd[2]
        __cse_13 = w[2]
        xbd[i_t] = xbd[i_t] + (__oldb_0 * __cse_13d + __cse_13 * __oldb_0d)
        xb[i_t] = xb[i_t] + __cse_13 * __oldb_0
        wbd[2] = wbd[2] + (__oldb_0 * __cse_10d + __cse_10 * __oldb_0d)
        wb[2] = wb[2] + __cse_10 * __oldb_0
        xbd[i_t + 1] = xbd[i_t + 1] + (__oldb_0 * __cse_11d + __cse_11 * __oldb_0d)
        xb[i_t + 1] = xb[i_t + 1] + __cse_11 * __oldb_0
        wbd[3] = wbd[3] + (__oldb_0 * __cse_12d + __cse_12 * __oldb_0d)
        wb[3] = wb[3] + __cse_12 * __oldb_0
        __oldb_0d = ybd[i_b + 1]
        __oldb_0 = yb[i_b + 1]
        ybd[i_b + 1] = 0.0
        yb[i_b + 1] = 0.0
        xbd[i_t] = xbd[i_t] + (__oldb_0 * __cse_9d + __cse_9 * __oldb_0d)
        xb[i_t] = xb[i_t] + __cse_9 * __oldb_0
        wbd[1] = wbd[1] + (__oldb_0 * __cse_10d + __cse_10 * __oldb_0d)
        wb[1] = wb[1] + __cse_10 * __oldb_0
        xbd[i_t + 1] = xbd[i_t + 1] + (__oldb_0 * __cse_13d + __cse_13 * __oldb_0d)
        xb[i_t + 1] = xb[i_t + 1] + __cse_13 * __oldb_0
        wbd[2] = wbd[2] + (__oldb_0 * __cse_12d + __cse_12 * __oldb_0d)
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
