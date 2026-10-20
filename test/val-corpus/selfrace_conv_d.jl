function selfrace_conv_d(x, xd, w, wd, z, zd, loss, lossd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            __cse_0 = w[ci]
            __cse_1 = x[(t + ci) - 1]
            ad = ad + (__cse_0 * xd[(t + ci) - 1] + __cse_1 * wd[ci])
            a = a + __cse_1 * __cse_0
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

function selfrace_conv(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            a = a + x[(t + ci) - 1] * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
