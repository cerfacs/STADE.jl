function cross_nested_d(x, xd, w, wd, z, zd, loss, lossd, i_n, i_nc)
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
            for cj = 1:2
                __cse_2 = w[cj]
                __cse_3 = x[(t + ci + cj) - 1]
                xvd = __cse_2 * xd[(t + ci + cj) - 1] + __cse_3 * wd[cj]
                xv = __cse_3 * __cse_2
                __cse_4 = xv * xvd
                ad = ad + (__cse_4 + __cse_4)
                a = a + xv * xv
            end
            xvd = 0.5 * xd[(t + ci) - 1]
            xv = x[(t + ci) - 1] * 0.5
            __cse_5 = w[ci + 1]
            __cse_6 = (xv * __cse_5) * xvd
            ad = ad + ((__cse_6 + __cse_6) + (xv * xv) * wd[ci + 1])
            a = a + xv * xv * __cse_5
        end
        zd[t] = ad
        z[t] = a
    end
    for m = 1:i_n
        __cse_7 = z[m]
        __cse_8 = __cse_7 * zd[m]
        lossd[1] = lossd[1] + (__cse_8 + __cse_8)
        loss[1] = loss[1] + __cse_7 * __cse_7
    end
    return nothing
end

function cross_nested(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv = x[(t + ci) - 1]
            a = a + xv * xv * w[ci]
            for cj = 1:2
                xv = x[(t + ci + cj) - 1] * w[cj]
                a = a + xv * xv
            end
            xv = x[(t + ci) - 1] * 0.5
            a = a + xv * xv * w[ci + 1]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
