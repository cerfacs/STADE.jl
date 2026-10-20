function if_scalar_rewrite_d(x, xd, w, wd, y, yd, loss, lossd, i_n)
    for t = 1:i_n
        wvd = 0.0
        wv = 0.0
        if x[t] > 0.0
            wvd = wd[t]
            wv = w[t]
        end
        __cse_0 = x[t]
        __cse_1 = (wv * __cse_0) * wvd
        __cse_2 = xd[t]
        yd[t] = (__cse_1 + __cse_1) + (wv * wv) * __cse_2
        y[t] = wv * wv * __cse_0
        wvd = wd[t + 1]
        wv = w[t + 1]
        __cse_3 = (wv * __cse_0) * wvd
        yd[t] = yd[t] + ((__cse_3 + __cse_3) + (wv * wv) * __cse_2)
        y[t] = y[t] + wv * wv * __cse_0
    end
    for m = 1:i_n
        __cse_4 = y[m]
        __cse_5 = __cse_4 * yd[m]
        lossd[1] = lossd[1] + (__cse_5 + __cse_5)
        loss[1] = loss[1] + __cse_4 * __cse_4
    end
    return nothing
end

function if_scalar_rewrite(x, w, y, loss, i_n)
    for t = 1:i_n
        wv = 0.0
        if x[t] > 0.0
            wv = w[t]
        end
        y[t] = wv * wv * x[t]
        wv = w[t + 1]
        y[t] = y[t] + wv * wv * x[t]
    end
    for m = 1:i_n
        loss[1] = loss[1] + y[m] * y[m]
    end
end
