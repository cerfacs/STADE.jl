function initstacks_red_refuse_invariant_b()
    return nothing
end

function red_refuse_invariant_b(loss, lossb, u, ub, c, cb, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * c[1]
    end
    for i_x = i_n:-1:1
        __cse_0 = lossb[1]
        ub[i_x] = ub[i_x] + c[1] * __cse_0
        cb[1] = cb[1] + u[i_x] * __cse_0
    end
    return nothing
end

function red_refuse_invariant(loss, u, c, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * c[1]
    end
end
