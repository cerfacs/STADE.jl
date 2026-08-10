import Pkg
haskey(Pkg.project().dependencies, "AMDGPU") || Pkg.add("AMDGPU")
using AMDGPU
AMDGPU.allowscalar(false)

function initstacks_branchsel_b_amdgpu()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function branchsel_b_amdgpu(loss, lossb, x, xb, y, yb, branch_stack)
    if x > y
        push!(branch_stack, 1)
        loss[1] = x ^ 2 - y
    else
        push!(branch_stack, 0)
        loss[1] = y ^ 2 - x
    end
    __branch = pop!(branch_stack)
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

function branchsel_amdgpu(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
    return nothing
end
