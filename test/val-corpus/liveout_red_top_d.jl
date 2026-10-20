function liveout_red_top_d(x, xd, w, wd, loss, lossd, i_nc)
    ad = 0.0
    a = 0.0
    for ci = 1:i_nc
        xv2d = xd[ci]
        xv2 = x[ci]
        __cse_0 = w[ci]
        __cse_1 = (xv2 * __cse_0) * xv2d
        ad = ad + ((__cse_1 + __cse_1) + (xv2 * xv2) * wd[ci])
        a = a + xv2 * xv2 * __cse_0
    end
    __cse_2 = a * ad
    lossd[1] = lossd[1] + (__cse_2 + __cse_2)
    loss[1] = loss[1] + a * a
    return nothing
end

function liveout_red_top(x, w, loss, i_nc)
    a = 0.0
    for ci = 1:i_nc
        xv2 = x[ci]
        a = a + xv2 * xv2 * w[ci]
    end
    loss[1] = loss[1] + a * a
end
