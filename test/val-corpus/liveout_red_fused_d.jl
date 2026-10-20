function liveout_red_fused_d(x, xd, w, wd, z, zd, i_n, i_nc)
    for t = 1:i_n
        ad = 0.0
        a = 0.0
        for ci = 1:i_nc
            xv2d = xd[(t + ci) - 1]
            xv2 = x[(t + ci) - 1]
            __cse_0 = w[ci]
            __cse_1 = (xv2 * __cse_0) * xv2d
            ad = ad + ((__cse_1 + __cse_1) + (xv2 * xv2) * wd[ci])
            a = a + xv2 * xv2 * __cse_0
        end
        zd[t] = ad
        z[t] = a
    end
    return nothing
end

function liveout_red_fused(x, w, z, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            xv2 = x[(t + ci) - 1]
            a = a + xv2 * xv2 * w[ci]
        end
        z[t] = a
    end
end
