import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function initstacks_branchsel_b_metal()
    branch_stack = Vector{Int64}()
    return branch_stack
end

function branchsel_hv_metal(loss, lossb, x, xb, y, yb, lossd, lossbd, xd, xbd, yd, ybd, branch_stack)
    if x > y
        push!(branch_stack, 1)
        lossd[1] = (2x) * xd + -yd
        loss[1] = x ^ 2 - y
    else
        push!(branch_stack, 0)
        lossd[1] = (2y) * yd + -xd
        loss[1] = y ^ 2 - x
    end
    __branch = pop!(branch_stack)
    if __branch == 1
        xbd = xbd + (lossb[1] * (2xd) + (2x) * lossbd[1])
        xb = xb + (2x) * lossb[1]
        ybd = ybd + -(lossbd[1])
        yb = yb + -(lossb[1])
        lossbd[1] = 0.0f0
        lossb[1] = 0.0f0
    else
        ybd = ybd + (lossb[1] * (2yd) + (2y) * lossbd[1])
        yb = yb + (2y) * lossb[1]
        xbd = xbd + -(lossbd[1])
        xb = xb + -(lossb[1])
        lossbd[1] = 0.0f0
        lossb[1] = 0.0f0
    end
    return (xb, xbd, yb, ybd)
end

function branchsel_metal(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
    return nothing
end
