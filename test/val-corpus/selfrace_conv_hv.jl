function initstacks_selfrace_conv_b()
    return nothing
end

function selfrace_conv_hv(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, zd, zbd, lossd, lossbd)
    a = 0.0
    ab = 0.0
    ad = 0.0
    abd = 0.0
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            __hcse_0 = w[ci]
            __hcse_1 = x[(t + ci) - 1]
            ad = ad + (__hcse_0 * xd[(t + ci) - 1] + __hcse_1 * wd[ci])
            a = a + __hcse_1 * __hcse_0
        end
        zd[t] = ad
        z[t] = a
    end
    for m = 1:i_n
        __cse_0d = zd[m]
        __cse_0 = z[m]
        __hcse_2 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_2 + __hcse_2)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for m = i_n:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = z[m]
        __cse_1d = __hcse_3 * zd[m] + __hcse_4 * lossbd[1]
        __cse_1 = __hcse_4 * __hcse_3
        zbd[m] = zbd[m] + __cse_1d
        zb[m] = zb[m] + __cse_1
        zbd[m] = zbd[m] + __cse_1d
        zb[m] = zb[m] + __cse_1
    end
    for t = i_n:-1:1
        __oldb_0d = zbd[t]
        __oldb_0 = zb[t]
        zbd[t] = 0.0
        zb[t] = 0.0
        abd = abd + __oldb_0d
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __hcse_5 = w[ci]
            xbd[(t + ci) - 1] = xbd[(t + ci) - 1] + (ab * wd[ci] + __hcse_5 * abd)
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __hcse_5 * ab
            __hcse_6 = x[(t + ci) - 1]
            wbd[ci] = wbd[ci] + (ab * xd[(t + ci) - 1] + __hcse_6 * abd)
            wb[ci] = wb[ci] + __hcse_6 * ab
        end
        abd = 0.0
        ab = 0.0
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
