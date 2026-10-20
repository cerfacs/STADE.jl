function so_carry_d(x, xd, w, wd, y, yd, loss, lossd, i_n, i_nc)
    rund = 0.0
    run = 0.0
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            wvd = wd[(ci - 1) * 3 + 1]
            wv = w[(ci - 1) * 3 + 1]
            __cse_0 = x[t]
            ad = ad + (wv * xd[t] + __cse_0 * wvd)
            a = a + __cse_0 * wv
            wvd = wd[(ci - 1) * 3 + 2]
            wv = w[(ci - 1) * 3 + 2]
            __cse_1 = x[t + 1]
            ad = ad + (wv * xd[t + 1] + __cse_1 * wvd)
            a = a + __cse_1 * wv
            wvd = wd[(ci - 1) * 3 + 3]
            wv = w[(ci - 1) * 3 + 3]
            __cse_2 = x[t + 2]
            ad = ad + (wv * xd[t + 2] + __cse_2 * wvd)
            a = a + __cse_2 * wv
        end
        rund = 0.5rund + ad
        run = run * 0.5 + a
        yd[t] = rund
        y[t] = run
    end
    for k = 1:i_n
        __cse_3 = y[k]
        __cse_4 = __cse_3 * yd[k]
        lossd[1] = lossd[1] + (__cse_4 + __cse_4)
        loss[1] = loss[1] + __cse_3 * __cse_3
    end
    return nothing
end

function so_carry(x, w, y, loss, i_n, i_nc)
    run = 0.0
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
        run = run * 0.5 + a
        y[t] = run
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
end
