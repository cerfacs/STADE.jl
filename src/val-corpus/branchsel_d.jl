function branchsel_d(loss, lossd, x, xd, y, yd)
    if x > y
        lossd[1] = (2x) * xd + -yd
        loss[1] = x ^ 2 - y
    else
        lossd[1] = (2y) * yd + -xd
        loss[1] = y ^ 2 - x
    end
    return nothing
end

function branchsel(loss, x, y)
    #= none:1 =#
    #= none:2 =#
    if x > y
        #= none:3 =#
        loss[1] = x ^ 2 - y
    else
        #= none:5 =#
        loss[1] = y ^ 2 - x
    end
end
