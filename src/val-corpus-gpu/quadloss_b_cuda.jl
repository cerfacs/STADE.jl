import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function initstacks_quadloss_b_cuda()
    return nothing
end

function quadloss_b_cuda(loss, lossb, x, xb, y, yb, z, zb)
    CUDA.@allowscalar begin
            loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
            xb = xb + (2x) * (y * lossb[1])
            yb = yb + x ^ 2 * lossb[1]
            yb = yb + (3.0z) * -(lossb[1])
            zb = zb + (3.0y) * -(lossb[1])
            zb = zb + (3 * z ^ 2) * lossb[1]
            lossb[1] = 0.0
        end
    return (xb, yb, zb)
end

function quadloss_cuda(loss, x, y, z)
    CUDA.@allowscalar begin
            loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
        end
    return nothing
end
