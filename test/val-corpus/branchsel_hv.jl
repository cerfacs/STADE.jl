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
        __cse_0d = lossbd[1]
        __cse_0 = lossb[1]
        __cse_2 = 2x
        xbd = xbd + (__cse_0 * (2xd) + __cse_2 * __cse_0d)
        xb = xb + __cse_2 * __cse_0
        ybd = ybd + -__cse_0d
        yb = yb + -__cse_0
        lossbd[1] = 0.0
        lossb[1] = 0.0
    else
        __cse_1d = lossbd[1]
        __cse_1 = lossb[1]
        __cse_3 = 2y
        ybd = ybd + (__cse_1 * (2yd) + __cse_3 * __cse_1d)
        yb = yb + __cse_3 * __cse_1
        xbd = xbd + -__cse_1d
        xb = xb + -__cse_1
        lossbd[1] = 0.0
        lossb[1] = 0.0
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
