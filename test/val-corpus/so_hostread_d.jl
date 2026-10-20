function so_hostread_d(x, xd, w, wd, y, yd, q, qd, loss, lossd, i_n, i_nc)
    wvd = 2.0 * wd[1]
    wv = w[1] * 2.0
    __cse_0 = wv * wvd
    qd[1] = __cse_0 + __cse_0
    q[1] = wv * wv
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            wvd = wd[(ci - 1) * 3 + 1]
            wv = w[(ci - 1) * 3 + 1]
            __cse_1 = x[t]
            ad = ad + (wv * xd[t] + __cse_1 * wvd)
            a = a + __cse_1 * wv
            wvd = wd[(ci - 1) * 3 + 2]
            wv = w[(ci - 1) * 3 + 2]
            __cse_2 = x[t + 1]
            ad = ad + (wv * xd[t + 1] + __cse_2 * wvd)
            a = a + __cse_2 * wv
            wvd = wd[(ci - 1) * 3 + 3]
            wv = w[(ci - 1) * 3 + 3]
            __cse_3 = x[t + 2]
            ad = ad + (wv * xd[t + 2] + __cse_3 * wvd)
            a = a + __cse_3 * wv
        end
        yd[t] = ad
        y[t] = a
    end
    for k = 1:i_n
        __cse_4 = y[k]
        __cse_5 = __cse_4 * yd[k]
        lossd[1] = (lossd[1] + (__cse_5 + __cse_5)) + qd[1]
        loss[1] = loss[1] + __cse_4 * __cse_4 + q[1]
    end
    return nothing
end

function so_hostread(x, w, y, q, loss, i_n, i_nc)
    wv = w[1] * 2.0
    q[1] = wv * wv
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[(ci - 1) * 3 + 1]
            a = a + x[t] * wv
            wv = w[(ci - 1) * 3 + 2]
            a = a + x[t + 1] * wv
            wv = w[(ci - 1) * 3 + 3]
            a = a + x[t + 2] * wv
        end
        y[t] = a
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k] + q[1]
    end
end
