function so_liveout_d(x, xd, w, wd, y, yd, z, zd, loss, lossd, i_nb, i_n, i_nc)
    for bb = 1:i_nb
        wvd = 0.0
        wv = 0.0
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
            yd[t] = ad
            y[t] = a
        end
        zd[bb] = 3.0wvd
        z[bb] = wv * 3.0
    end
    for k = 1:i_n
        __cse_3 = y[k]
        __cse_4 = __cse_3 * yd[k]
        lossd[1] = lossd[1] + (__cse_4 + __cse_4)
        loss[1] = loss[1] + __cse_3 * __cse_3
    end
    for k = 1:i_nb
        __cse_5 = z[k]
        __cse_6 = __cse_5 * zd[k]
        lossd[1] = lossd[1] + (__cse_6 + __cse_6)
        loss[1] = loss[1] + __cse_5 * __cse_5
    end
    return nothing
end

function so_liveout(x, w, y, z, loss, i_nb, i_n, i_nc)
    for bb = 1:i_nb
        wv = 0.0
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
        z[bb] = wv * 3.0
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
    for k = 1:i_nb
        loss[1] = loss[1] + z[k] * z[k]
    end
end
