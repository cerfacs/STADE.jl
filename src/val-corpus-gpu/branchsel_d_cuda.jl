import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function branchsel_d_cuda(loss, lossd, x, xd, y, yd)
    if x > y
        lossd[1] = (2x) * xd + -yd
        loss[1] = x ^ 2 - y
    else
        lossd[1] = (2y) * yd + -xd
        loss[1] = y ^ 2 - x
    end
    return nothing
end

function branchsel_cuda(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
    return nothing
end
