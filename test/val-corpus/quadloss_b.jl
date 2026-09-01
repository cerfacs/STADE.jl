function initstacks_quadloss_b()
    return nothing
end

function quadloss_b(loss, lossb, x, xb, y, yb, z, zb)
    __cse_0 = x ^ 2
    loss[1] = (__cse_0 * y - 3.0 * y * z) + z ^ 3
    __cse_1 = lossb[1]
    xb = xb + (2x) * (y * __cse_1)
    yb = yb + __cse_0 * __cse_1
    __cse_2 = -__cse_1
    yb = yb + (3.0z) * __cse_2
    zb = zb + (3.0y) * __cse_2
    zb = zb + (3 * z ^ 2) * __cse_1
    lossb[1] = 0.0
    return (xb, yb, zb)
end

function quadloss(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
end
