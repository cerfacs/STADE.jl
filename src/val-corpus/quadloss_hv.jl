function initstacks_quadloss_b()
    return nothing
end

function quadloss_hv(loss, lossb, x, xb, y, yb, z, zb, lossd, lossbd, xd, xbd, yd, ybd, zd, zbd)
    lossd[1] = ((y * ((2x) * xd) + x ^ 2 * yd) + -(((3.0z) * yd + (3.0y) * zd))) + (3 * z ^ 2) * zd
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
    xbd = xbd + ((y * lossb[1]) * (2xd) + (2x) * (lossb[1] * yd + y * lossbd[1]))
    xb = xb + (2x) * (y * lossb[1])
    ybd = ybd + (lossb[1] * ((2x) * xd) + x ^ 2 * lossbd[1])
    yb = yb + x ^ 2 * lossb[1]
    ybd = ybd + (-(lossb[1]) * (3.0zd) + (3.0z) * -(lossbd[1]))
    yb = yb + (3.0z) * -(lossb[1])
    zbd = zbd + (-(lossb[1]) * (3.0yd) + (3.0y) * -(lossbd[1]))
    zb = zb + (3.0y) * -(lossb[1])
    zbd = zbd + (lossb[1] * (3 * ((2z) * zd)) + (3 * z ^ 2) * lossbd[1])
    zb = zb + (3 * z ^ 2) * lossb[1]
    lossbd[1] = 0.0
    lossb[1] = 0.0
    return (xb, xbd, yb, ybd, zb, zbd)
end

function quadloss(loss, x, y, z)
    #= none:1 =#
    #= none:2 =#
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
end
