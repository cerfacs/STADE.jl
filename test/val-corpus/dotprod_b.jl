function initstacks_dotprod_b()
    return nothing
end

function dotprod_b(loss, lossb, u, ub, v, vb, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
    for i_x = i_n:-1:1
        ub[i_x] = ub[i_x] + v[i_x] * lossb[1]
        vb[i_x] = vb[i_x] + u[i_x] * lossb[1]
    end
    return nothing
end

function dotprod(loss, u, v, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
end
