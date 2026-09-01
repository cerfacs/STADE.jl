function initstacks_quadloss_b()
    return nothing
end

function quadloss_hv(loss, lossb, x, xb, y, yb, z, zb, lossd, lossbd, xd, xbd, yd, ybd, zd, zbd)
    __cse_2 = 2x
    __cse_3 = __cse_2 * xd
    __cse_4 = x ^ 2
    __cse_5 = 3.0z
    __cse_6 = 3.0y
    __cse_7 = 3 * z ^ 2
    lossd[1] = ((y * __cse_3 + __cse_4 * yd) + -((__cse_5 * yd + __cse_6 * zd))) + __cse_7 * zd
    loss[1] = (__cse_4 * y - 3.0 * y * z) + z ^ 3
    __cse_0d = lossbd[1]
    __cse_0 = lossb[1]
    __cse_8 = y * __cse_0
    xbd = xbd + (__cse_8 * (2xd) + __cse_2 * (__cse_0 * yd + y * __cse_0d))
    xb = xb + __cse_2 * __cse_8
    ybd = ybd + (__cse_0 * __cse_3 + __cse_4 * __cse_0d)
    yb = yb + __cse_4 * __cse_0
    __cse_1d = -__cse_0d
    __cse_1 = -__cse_0
    ybd = ybd + (__cse_1 * (3.0zd) + __cse_5 * __cse_1d)
    yb = yb + __cse_5 * __cse_1
    zbd = zbd + (__cse_1 * (3.0yd) + __cse_6 * __cse_1d)
    zb = zb + __cse_6 * __cse_1
    zbd = zbd + (__cse_0 * (3 * ((2z) * zd)) + __cse_7 * __cse_0d)
    zb = zb + __cse_7 * __cse_0
    lossbd[1] = 0.0
    lossb[1] = 0.0
    return (xb, xbd, yb, ybd, zb, zbd)
end

function quadloss(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
end
