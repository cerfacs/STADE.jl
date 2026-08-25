function initstacks_pipeline_b()
    return nothing
end

function pipeline_hv(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd)
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
    for i_x2 = i_n:-1:1
        wbd[i_x2] = wbd[i_x2] + lossbd[1]
        wb[i_x2] = wb[i_x2] + lossb[1]
    end
    for i_x = i_n:-1:1
        vbd[i_x] = vbd[i_x] + (wb[i_x] * ud[i_x] + u[i_x] * wbd[i_x])
        vb[i_x] = vb[i_x] + u[i_x] * wb[i_x]
        ubd[i_x] = ubd[i_x] + (wb[i_x] * vd[i_x] + v[i_x] * wbd[i_x])
        ub[i_x] = ub[i_x] + v[i_x] * wb[i_x]
        wbd[i_x] = 0.0
        wb[i_x] = 0.0
    end
    for i_x = i_n:-1:1
        ubd[i_x] = ubd[i_x] + (vb[i_x] * (2 * ud[i_x]) + (2 * u[i_x]) * vbd[i_x])
        ub[i_x] = ub[i_x] + (2 * u[i_x]) * vb[i_x]
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
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
