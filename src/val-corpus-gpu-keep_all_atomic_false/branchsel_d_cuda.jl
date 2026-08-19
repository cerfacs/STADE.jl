import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function branchsel_d_cuda(loss, lossd, x, xd, y, yd)
    if x > y
        CUDA.@allowscalar begin
                lossd[1] = (2x) * xd + -yd
                loss[1] = x ^ 2 - y
            end
    else
        CUDA.@allowscalar begin
                lossd[1] = (2y) * yd + -xd
                loss[1] = y ^ 2 - x
            end
    end
    return nothing
end

function branchsel_cuda(loss, x, y)
    if x > y
        CUDA.@allowscalar begin
                loss[1] = x ^ 2 - y
            end
    else
        CUDA.@allowscalar begin
                loss[1] = y ^ 2 - x
            end
    end
    return nothing
end
