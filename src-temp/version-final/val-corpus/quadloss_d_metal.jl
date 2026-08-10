import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function quadloss_d_metal(loss, lossd, x, xd, y, yd, z, zd)
    lossd[1] = ((y * ((2x) * xd) + x ^ 2 * yd) + -(((3.0f0z) * yd + (3.0f0y) * zd))) + (3 * z ^ 2) * zd
    loss[1] = (x ^ 2 * y - 3.0f0 * y * z) + z ^ 3
    return nothing
end

function quadloss_metal(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0f0 * y * z) + z ^ 3
    return nothing
end
