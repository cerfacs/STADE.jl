function cross_ifwrite_d(x, xd, w, wd, z, zd, loss, lossd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            __cse_0 = w[ci]
            __cse_1 = (xv * __cse_0) * xvd
            ad = ad + ((__cse_1 + __cse_1) + (xv * xv) * wd[ci])
            a = a + xv * xv * __cse_0
            if x[t + ci] > 0.5
                xvd = 2.0xvd
                xv = xv * 2.0
            end
            __cse_2 = xv * xvd
            ad = ad + (__cse_2 + __cse_2)
            a = a + xv * xv
        end
        zd[t] = ad
        z[t] = a
    end
    for m = 1:i_n
        __cse_3 = z[m]
        __cse_4 = __cse_3 * zd[m]
        lossd[1] = lossd[1] + (__cse_4 + __cse_4)
        loss[1] = loss[1] + __cse_3 * __cse_3
    end
    return nothing
end

function cross_ifwrite(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
            if x[t + ci] > 0.5
                xv = xv * 2.0
            end
            a = a + xv * xv
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
