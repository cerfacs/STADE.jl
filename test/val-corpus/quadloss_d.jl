function quadloss_d(loss, lossd, x, xd, y, yd, z, zd)
    __cse_0 = x ^ 2
    lossd[1] = ((y * ((2x) * xd) + __cse_0 * yd) + -(((3.0z) * yd + (3.0y) * zd))) + (3 * z ^ 2) * zd
    loss[1] = (__cse_0 * y - 3.0 * y * z) + z ^ 3
    return nothing
end

function quadloss(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
end
