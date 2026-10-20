function so_inner_d(x, xd, w, wd, q, qd, y, yd, loss, lossd, i_n, i_nc)
    rund = 0.0
    run = 0.0
    for t = 1:i_n
        for ci = 1:i_nc
            xvd = xd[ci]
            xv = x[ci]
            wvd = wd[ci]
            wv = w[ci]
            qd[(t - 1) * i_nc + ci] = wv * xvd + xv * wvd
            q[(t - 1) * i_nc + ci] = xv * wv
            __cse_0 = 1.0 + run
            __cse_1 = x[ci + 1]
            xvd = __cse_0 * xd[ci + 1] + __cse_1 * rund
            xv = __cse_1 * __cse_0
            __cse_2 = w[ci + 1]
            wvd = __cse_0 * wd[ci + 1] + __cse_2 * rund
            wv = __cse_2 * __cse_0
            qd[(t - 1) * i_nc + ci] = qd[(t - 1) * i_nc + ci] + (((wv * run) * xvd + (xv * run) * wvd) + (xv * wv) * rund)
            q[(t - 1) * i_nc + ci] = q[(t - 1) * i_nc + ci] + xv * wv * run
        end
        rund = 0.5rund + qd[(t - 1) * i_nc + 1]
        run = run * 0.5 + q[(t - 1) * i_nc + 1]
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

function so_inner(x, w, q, y, loss, i_n, i_nc)
    run = 0.0
    for t = 1:i_n
        for ci = 1:i_nc
            xv = x[ci]
            wv = w[ci]
            q[(t - 1) * i_nc + ci] = xv * wv
            xv = x[ci + 1] * (1.0 + run)
            wv = w[ci + 1] * (1.0 + run)
            q[(t - 1) * i_nc + ci] = q[(t - 1) * i_nc + ci] + xv * wv * run
        end
        run = run * 0.5 + q[(t - 1) * i_nc + 1]
        y[t] = run
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
end
