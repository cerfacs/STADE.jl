function pipeline_d(loss, lossd, u, ud, v, vd, w, wd, i_n)
    for i_x = 1:i_n
        vd[i_x] = (2 * u[i_x]) * ud[i_x]
        v[i_x] = u[i_x] ^ 2 + 1.0
    end
    for i_x = 1:i_n
        wd[i_x] = u[i_x] * vd[i_x] + v[i_x] * ud[i_x]
        w[i_x] = v[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        lossd[1] = lossd[1] + wd[i_x2]
        loss[1] = loss[1] + w[i_x2]
    end
    return nothing
end

function pipeline(loss, u, v, w, i_n)
    for i_x = 1:i_n
        v[i_x] = u[i_x] ^ 2 + 1.0
    end
    for i_x = 1:i_n
        w[i_x] = v[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + w[i_x2]
    end
end
