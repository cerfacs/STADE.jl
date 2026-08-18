import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function initstacks_branchsel_b_cuda()
    branch_stack = CuArray{Int64}(undef, 1)
    return branch_stack
end

function branchsel_b_cuda(loss, lossb, x, xb, y, yb, branch_stack)
    if x > y
        branch_stack[1] = 1
        loss[1] = x ^ 2 - y
    else
        branch_stack[1] = 0
        loss[1] = y ^ 2 - x
    end
    __branch = branch_stack[1]
    if __branch == 1
        xb = xb + (2x) * lossb[1]
        yb = yb + -(lossb[1])
        lossb[1] = 0.0
    else
        yb = yb + (2y) * lossb[1]
        xb = xb + -(lossb[1])
        lossb[1] = 0.0
    end
    return (xb, yb)
end

function branchsel_cuda(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
    return nothing
end
