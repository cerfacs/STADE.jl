import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function quadloss_d_cuda(loss, lossd, x, xd, y, yd, z, zd)
    CUDA.@allowscalar begin
            lossd[1] = ((y * ((2x) * xd) + x ^ 2 * yd) + -(((3.0z) * yd + (3.0y) * zd))) + (3 * z ^ 2) * zd
            loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
        end
    return nothing
end

function quadloss_cuda(loss, x, y, z)
    CUDA.@allowscalar begin
            loss[1] = (x ^ 2 * y - 3.0 * y * z) + z ^ 3
        end
    return nothing
end
