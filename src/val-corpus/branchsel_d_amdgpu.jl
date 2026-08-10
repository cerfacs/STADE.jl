import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function branchsel_d_amdgpu(loss, lossd, x, xd, y, yd)
    if x > y
        lossd[1] = (2x) * xd + -yd
        loss[1] = x ^ 2 - y
    else
        lossd[1] = (2y) * yd + -xd
        loss[1] = y ^ 2 - x
    end
    return nothing
end

function branchsel_amdgpu(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
    return nothing
end
