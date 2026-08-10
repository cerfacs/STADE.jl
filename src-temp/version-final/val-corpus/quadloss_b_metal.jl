import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function initstacks_quadloss_b_metal()
    return nothing
end

function quadloss_b_metal(loss, lossb, x, xb, y, yb, z, zb)
    loss[1] = (x ^ 2 * y - 3.0f0 * y * z) + z ^ 3
    xb = xb + (2x) * (y * lossb[1])
    yb = yb + x ^ 2 * lossb[1]
    yb = yb + (3.0f0z) * -(lossb[1])
    zb = zb + (3.0f0y) * -(lossb[1])
    zb = zb + (3 * z ^ 2) * lossb[1]
    lossb[1] = 0.0f0
    return (xb, yb, zb)
end

function quadloss_metal(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0f0 * y * z) + z ^ 3
    return nothing
end
