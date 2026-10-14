function initstacks_red_refuse_barevar_b()
    return nothing
end

function red_refuse_barevar_b(loss, lossb, u, ub, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * i_x
    end
    for i_x = i_n:-1:1
        ub[i_x] = ub[i_x] + i_x * lossb[1]
    end
    return nothing
end

function red_refuse_barevar(loss, u, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * i_x
    end
end
