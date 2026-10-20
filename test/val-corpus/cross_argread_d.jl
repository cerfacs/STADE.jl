function cross_argread_d(x, xd, w, wd, loss, lossd, s, sd, i_nc)
    __cse_0 = s * sd
    ad = __cse_0 + __cse_0
    a = s * s
    for ci = 1:i_nc
        sd = 2.0 * xd[ci]
        s = x[ci] * 2.0
        __cse_1 = w[ci]
        __cse_2 = (s * __cse_1) * sd
        ad = ad + ((__cse_2 + __cse_2) + (s * s) * wd[ci])
        a = a + s * s * __cse_1
    end
    __cse_3 = a * ad
    lossd[1] = lossd[1] + (__cse_3 + __cse_3)
    loss[1] = loss[1] + a * a
    return sd
end

function cross_argread(x, w, loss, s, i_nc)
    a = s * s
    for ci = 1:i_nc
        s = x[ci] * 2.0
        a = a + s * s * w[ci]
    end
    loss[1] = loss[1] + a * a
end
