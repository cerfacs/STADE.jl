function so_tile_d(x, xd, w, wd, b, bd, y, yd, z, zd, loss, lossd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            wvd = wd[(ci - 1) * 2 + 1]
            wv = w[(ci - 1) * 2 + 1]
            ad = ad + (wv * xvd + xv * wvd)
            a = a + xv * wv
            xvd = xd[t + ci]
            xv = x[t + ci]
            wvd = wd[(ci - 1) * 2 + 2]
            wv = w[(ci - 1) * 2 + 2]
            ad = ad + (wv * xvd + xv * wvd)
            a = a + xv * wv
        end
        __cse_0 = a + b[1]
        yd[t] = (0.5 * (1.0 + sign(__cse_0))) * (ad + bd[1])
        y[t] = max(__cse_0, 0.0)
    end
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xvd = yd[(t + ci) - 1]
            xv = y[(t + ci) - 1]
            wvd = wd[(ci - 1) * 2 + 2]
            wv = w[(ci - 1) * 2 + 2]
            ad = ad + (wv * xvd + xv * wvd)
            a = a + xv * wv
            xvd = yd[t + ci]
            xv = y[t + ci]
            wvd = wd[(ci - 1) * 2 + 1]
            wv = w[(ci - 1) * 2 + 1]
            ad = ad + (wv * xvd + xv * wvd)
            a = a + xv * wv
        end
        zd[t] = ad
        z[t] = a
    end
    for k = 1:i_n
        __cse_1 = z[k]
        __cse_2 = __cse_1 * zd[k]
        lossd[1] = lossd[1] + (__cse_2 + __cse_2)
        loss[1] = loss[1] + __cse_1 * __cse_1
    end
    return nothing
end

function so_tile(x, w, b, y, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
            xv = x[t + ci]
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
        end
        y[t] = max(a + b[1], 0.0)
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = y[(t + ci) - 1]
            wv = w[(ci - 1) * 2 + 2]
            a = a + xv * wv
            xv = y[t + ci]
            wv = w[(ci - 1) * 2 + 1]
            a = a + xv * wv
        end
        z[t] = a
    end
    for k = 1:i_n
        loss[1] = loss[1] + z[k] * z[k]
    end
end
