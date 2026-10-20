function cross_mixed_d(x, xd, w, wd, loss, lossd, i_nc)
    ad = 0.0
    a = 0.0
    for ci = 1:i_nc
        __cse_0 = w[ci]
        __cse_1 = x[ci]
        __cse_2 = wd[ci]
        tmpd = __cse_0 * xd[ci] + __cse_1 * __cse_2
        tmp = __cse_1 * __cse_0
        xvd = 0.5tmpd
        xv = tmp * 0.5
        __cse_3 = (xv * __cse_0) * xvd
        ad = ad + ((__cse_3 + __cse_3) + (xv * xv) * __cse_2)
        a = a + xv * xv * __cse_0
    end
    bd = 0.0
    b = 0.0
    for ci = 1:i_nc
        xvd = xd[ci]
        xv = x[ci]
        __cse_4 = w[ci]
        __cse_5 = (xv * __cse_4) * xvd
        bd = bd + ((__cse_5 + __cse_5) + (xv * xv) * wd[ci])
        b = b + xv * xv * __cse_4
    end
    __cse_6 = a * ad
    __cse_7 = b * bd
    lossd[1] = (lossd[1] + (__cse_6 + __cse_6)) + (__cse_7 + __cse_7)
    loss[1] = loss[1] + a * a + b * b
    return nothing
end

function cross_mixed(x, w, loss, i_nc)
    a = 0.0
    for ci = 1:i_nc
        tmp = x[ci] * w[ci]
        xv = tmp * 0.5
        a = a + xv * xv * w[ci]
    end
    b = 0.0
    for ci = 1:i_nc
        xv = x[ci]
        b = b + xv * xv * w[ci]
    end
    loss[1] = loss[1] + a * a + b * b
end
