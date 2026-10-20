function cross_readbefore_d(x, xd, w, wd, z, zd, loss, lossd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        xvd = 0.5 * xd[t]
        xv = x[t] * 0.5
        for ci = 1:i_nc
            __cse_0 = w[ci]
            __cse_1 = (xv * __cse_0) * xvd
            ad = ad + ((__cse_1 + __cse_1) + (xv * xv) * wd[ci])
            a = a + xv * xv * __cse_0
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
        end
        zd[t] = ad
        z[t] = a
    end
    for m = 1:i_n
        __cse_2 = z[m]
        __cse_3 = __cse_2 * zd[m]
        lossd[1] = lossd[1] + (__cse_3 + __cse_3)
        loss[1] = loss[1] + __cse_2 * __cse_2
    end
    return nothing
end

function cross_readbefore(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        xv = x[t] * 0.5
        for ci = 1:i_nc
            a = a + xv * xv * w[ci]
            xv = x[(t + ci) - 1]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
