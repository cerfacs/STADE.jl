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
        __oldb_0 = lossb[1]
        lossb[1] = 0.0
        xb = xb + (2x) * __oldb_0
        yb = yb + -__oldb_0
    else
        __oldb_0 = lossb[1]
        lossb[1] = 0.0
        yb = yb + (2y) * __oldb_0
        xb = xb + -__oldb_0
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
