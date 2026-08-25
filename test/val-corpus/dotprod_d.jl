function dotprod_d(loss, lossd, u, ud, v, vd, i_n)
    for i_x = 1:i_n
        lossd[1] = lossd[1] + (v[i_x] * ud[i_x] + u[i_x] * vd[i_x])
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
    return nothing
end

function dotprod(loss, u, v, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
end
