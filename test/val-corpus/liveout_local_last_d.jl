function liveout_local_last_d(x, xd, w, wd, z, zd, loss, lossd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xv2d = xd[(t + ci) - 1]
            xv2 = x[(t + ci) - 1]
            __cse_0 = w[ci]
            __cse_1 = (xv2 * __cse_0) * xv2d
            ad = (__cse_1 + __cse_1) + (xv2 * xv2) * wd[ci]
            a = xv2 * xv2 * __cse_0
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

function liveout_local_last(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[(t + ci) - 1]
            a = xv2 * xv2 * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
