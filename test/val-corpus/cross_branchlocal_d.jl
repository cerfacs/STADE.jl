function cross_branchlocal_d(x, xd, w, wd, z, zd, loss, lossd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            if x[(t + ci) - 1] > 0.5
                xvd = 2.0 * xd[(t + ci) - 1]
                xv = x[(t + ci) - 1] * 2.0
                __cse_0 = w[ci]
                __cse_1 = (xv * __cse_0) * xvd
                ad = ad + ((__cse_1 + __cse_1) + (xv * xv) * wd[ci])
                a = a + xv * xv * __cse_0
            end
        end
        zd[t] = ad
        z[t] = a
    end
    for t = 1:i_n
        bd = 0.0
        b = 0.0
        for ci = 1:i_nc
            xvd = xd[(t + ci) - 1]
            xv = x[(t + ci) - 1]
            __cse_2 = w[ci]
            __cse_3 = (xv * __cse_2) * xvd
            bd = bd + ((__cse_3 + __cse_3) + (xv * xv) * wd[ci])
            b = b + xv * xv * __cse_2
        end
        zd[i_n + t] = bd
        z[i_n + t] = b
    end
    for m = 1:2i_n
        __cse_4 = z[m]
        __cse_5 = __cse_4 * zd[m]
        lossd[1] = lossd[1] + (__cse_5 + __cse_5)
        loss[1] = loss[1] + __cse_4 * __cse_4
    end
    return nothing
end

function cross_branchlocal(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            if x[(t + ci) - 1] > 0.5
                xv = x[(t + ci) - 1] * 2.0
                a = a + xv * xv * w[ci]
            end
        end
        z[t] = a
    end
    for t = 1:i_n
        b = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            b = b + xv * xv * w[ci]
        end
        z[i_n + t] = b
    end
    for m = 1:2i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
