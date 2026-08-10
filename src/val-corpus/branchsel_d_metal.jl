import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function branchsel_d_metal(loss, lossd, x, xd, y, yd)
    if x > y
        lossd[1] = (2x) * xd + -yd
        loss[1] = x ^ 2 - y
    else
        lossd[1] = (2y) * yd + -xd
        loss[1] = y ^ 2 - x
    end
    return nothing
end

function branchsel_metal(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
    return nothing
end
