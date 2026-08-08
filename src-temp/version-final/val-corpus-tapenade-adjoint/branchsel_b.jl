function initstacks_branchsel_b()
    return
end
function branchsel_b(loss, lossb, x, xb, y, yb)
    if x > y
        xb = xb + 2 * x * lossb[1]
        yb = yb - lossb[1]
        lossb[1] = 0.0
    else
        yb = yb + 2 * y * lossb[1]
        xb = xb - lossb[1]
        lossb[1] = 0.0
    end
    return xb,yb
end
function branchsel(loss, x, y)
    if x > y
        loss[1] = x ^ 2 - y
    else
        loss[1] = y ^ 2 - x
    end
end
