function initstacks_quadloss_b()
    return nothing
end

function quadloss_hv(loss, lossb, x, xb, y, yb, z, zb, lossd, lossbd, xd, xbd, yd, ybd, zd, zbd)
    __hcse_0 = 2x
    __hcse_1 = __hcse_0 * xd
    __hcse_2 = x ^ 2
    __hcse_3 = 3.0z
    __hcse_4 = 3.0y
    __hcse_5 = 3 * z ^ 2
    lossd[1] = ((y * __hcse_1 + __hcse_2 * yd) + -((__hcse_3 * yd + __hcse_4 * zd))) + __hcse_5 * zd
    loss[1] = (__hcse_2 * y - 3.0 * y * z) + z ^ 3
    __oldb_0d = lossbd[1]
    __oldb_0 = lossb[1]
    lossbd[1] = 0.0
    lossb[1] = 0.0
    __hcse_6 = y * __oldb_0
    xbd = xbd + (__hcse_6 * (2xd) + __hcse_0 * (__oldb_0 * yd + y * __oldb_0d))
    xb = xb + __hcse_0 * __hcse_6
    ybd = ybd + (__oldb_0 * __hcse_1 + __hcse_2 * __oldb_0d)
    yb = yb + __hcse_2 * __oldb_0
    __cse_0d = -__oldb_0d
    __cse_0 = -__oldb_0
    ybd = ybd + (__cse_0 * (3.0zd) + __hcse_3 * __cse_0d)
    yb = yb + __hcse_3 * __cse_0
    zbd = zbd + (__cse_0 * (3.0yd) + __hcse_4 * __cse_0d)
    zb = zb + __hcse_4 * __cse_0
    zbd = zbd + (__oldb_0 * (3 * ((2z) * zd)) + __hcse_5 * __oldb_0d)
    zb = zb + __hcse_5 * __oldb_0
    return (xb, xbd, yb, ybd, zb, zbd)
end

function quadloss(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
end
