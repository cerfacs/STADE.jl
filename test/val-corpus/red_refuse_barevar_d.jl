function red_refuse_barevar_d(loss, lossd, u, ud, i_n)
    for i_x = 1:i_n
        lossd[1] = lossd[1] + i_x * ud[i_x]
        loss[1] = loss[1] + u[i_x] * i_x
    end
    return nothing
end

function red_refuse_barevar(loss, u, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * i_x
    end
end
