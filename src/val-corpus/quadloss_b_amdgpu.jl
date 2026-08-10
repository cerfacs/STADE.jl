import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function initstacks_quadloss_b_amdgpu()
    return nothing
end

function quadloss_b_amdgpu(loss, lossb, x, xb, y, yb, z, zb)
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
    xb = xb + (2x) * (y * lossb[1])
    yb = yb + x ^ 2 * lossb[1]
    yb = yb + (3.0z) * -(lossb[1])
    zb = zb + (3.0y) * -(lossb[1])
    zb = zb + (3 * z ^ 2) * lossb[1]
    lossb[1] = 0.0
    return (xb, yb, zb)
end

function quadloss_amdgpu(loss, x, y, z)
    loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
    return nothing
end
