function initstacks_branchsel_b()
    branch_stack = Vector{Int64}(undef, 1)
    return branch_stack
end

function branchsel_hv(loss, lossb, x, xb, y, yb, lossd, lossbd, xd, xbd, yd, ybd, branch_stack)
    if x > y
        branch_stack[1] = 1
        lossd[1] = (2x) * xd + -yd
        loss[1] = x ^ 2 - y
    else
        branch_stack[1] = 0
        lossd[1] = (2y) * yd + -xd
        loss[1] = y ^ 2 - x
    end
    __branch = branch_stack[1]
    if __branch == 1
        __oldb_0d = lossbd[1]
        __oldb_0 = lossb[1]
        lossbd[1] = 0.0
        lossb[1] = 0.0
        __hcse_0 = 2x
        xbd = xbd + (__oldb_0 * (2xd) + __hcse_0 * __oldb_0d)
        xb = xb + __hcse_0 * __oldb_0
        ybd = ybd + -__oldb_0d
        yb = yb + -__oldb_0
    else
        __oldb_0d = lossbd[1]
        __oldb_0 = lossb[1]
        lossbd[1] = 0.0
        lossb[1] = 0.0
        __hcse_1 = 2y
        ybd = ybd + (__oldb_0 * (2yd) + __hcse_1 * __oldb_0d)
        yb = yb + __hcse_1 * __oldb_0
        xbd = xbd + -__oldb_0d
        xb = xb + -__oldb_0
    end
    return (xb, xbd, yb, ybd)
end

function branchsel(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
end
