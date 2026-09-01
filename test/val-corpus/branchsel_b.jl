function initstacks_branchsel_b()
    branch_stack = Vector{Int64}(undef, 1)
    return branch_stack
end

function branchsel_b(loss, lossb, x, xb, y, yb, branch_stack)
    if x > y
        branch_stack[1] = 1
        loss[1] = x ^ 2 - y
    else
        branch_stack[1] = 0
        loss[1] = y ^ 2 - x
    end
    __branch = branch_stack[1]
    if __branch == 1
        __cse_0 = lossb[1]
        xb = xb + (2x) * __cse_0
        yb = yb + -__cse_0
        lossb[1] = 0.0
    else
        __cse_1 = lossb[1]
        yb = yb + (2y) * __cse_1
        xb = xb + -__cse_1
        lossb[1] = 0.0
    end
    return (xb, yb)
end

function branchsel(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
end
