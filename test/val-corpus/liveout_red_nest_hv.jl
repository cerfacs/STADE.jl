function initstacks_liveout_red_nest_b(i_n)
    return nothing
end

function liveout_red_nest_hv(x, xb, w, wb, z, zb, loss, lossb, i_n, i_nc, xd, xbd, wd, wbd, zd, zbd, lossd, lossbd)
    a = 0.0
    xv2 = 0.0
    ab = 0.0
    xv2b = 0.0
    ad = 0.0
    abd = 0.0
    xv2d = 0.0
    xv2bd = 0.0
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xv2d = xd[(t + ci) - 1]
            xv2 = x[(t + ci) - 1]
            __hcse_0 = w[ci]
            __hcse_1 = (xv2 * __hcse_0) * xv2d
            ad = ad + ((__hcse_1 + __hcse_1) + (xv2 * xv2) * wd[ci])
            a = a + xv2 * xv2 * __hcse_0
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
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xv2d = xd[(t + ci) - 1]
            xv2 = x[(t + ci) - 1]
            __hcse_5 = w[ci]
            __hcse_6 = (xv2 * __hcse_5) * xv2d
            ad = ad + ((__hcse_6 + __hcse_6) + (xv2 * xv2) * wd[ci])
            a = a + xv2 * xv2 * __hcse_5
        end
        __oldb_0d = zbd[t]
        __oldb_0 = zb[t]
        zbd[t] = 0.0
        zb[t] = 0.0
        abd = abd + __oldb_0d
        ab = ab + __oldb_0
        for ci = 1:i_nc
            __cse_2d = xd[(t + ci) - 1]
            __cse_2 = x[(t + ci) - 1]
            xv2d = __cse_2d
            xv2 = __cse_2
            __cse_3d = wd[ci]
            __cse_3 = w[ci]
            __hcse_7 = (xv2 * __cse_3) * xv2d
            ad = ad + ((__hcse_7 + __hcse_7) + (xv2 * xv2) * __cse_3d)
            a = a + xv2 * xv2 * __cse_3
            xv2d = __cse_2d
            xv2 = __cse_2
            __hcse_8 = xv2 * __cse_3
            __cse_4d = ab * (__cse_3 * xv2d + xv2 * __cse_3d) + __hcse_8 * abd
            __cse_4 = __hcse_8 * ab
            xv2bd = xv2bd + __cse_4d
            xv2b = xv2b + __cse_4
            xv2bd = xv2bd + __cse_4d
            xv2b = xv2b + __cse_4
            __hcse_9 = xv2 * xv2d
            __hcse_10 = xv2 * xv2
            wbd[ci] = wbd[ci] + (ab * (__hcse_9 + __hcse_9) + __hcse_10 * abd)
            wb[ci] = wb[ci] + __hcse_10 * ab
            __oldb_0d = xv2bd
            __oldb_0 = xv2b
            xv2bd = 0.0
            xv2b = 0.0
            xbd[(t + ci) - 1] = xbd[(t + ci) - 1] + __oldb_0d
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + __oldb_0
        end
        abd = 0.0
        ab = 0.0
    end
    return nothing
end

function liveout_red_nest(x, w, z, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[(t + ci) - 1]
            a = a + xv2 * xv2 * w[ci]
        end
        z[t] = a
    end
    for m = 1:i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
