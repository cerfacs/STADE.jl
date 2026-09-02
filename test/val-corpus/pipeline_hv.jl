function initstacks_pipeline_b()
    return nothing
end

function pipeline_hv(loss, lossb, u, ub, v, vb, w, wb, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd)
    for i_x = 1:i_n
        __hcse_0 = u[i_x]
        vd[i_x] = (2__hcse_0) * ud[i_x]
        v[i_x] = __hcse_0 ^ 2 + 1.0
    end
    for i_x = 1:i_n
        __hcse_1 = u[i_x]
        __hcse_2 = v[i_x]
        wd[i_x] = __hcse_1 * vd[i_x] + __hcse_2 * ud[i_x]
        w[i_x] = __hcse_2 * __hcse_1
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
        __oldb_0d = wbd[i_x]
        __oldb_0 = wb[i_x]
        wbd[i_x] = 0.0
        wb[i_x] = 0.0
        __hcse_3 = u[i_x]
        vbd[i_x] = vbd[i_x] + (__oldb_0 * ud[i_x] + __hcse_3 * __oldb_0d)
        vb[i_x] = vb[i_x] + __hcse_3 * __oldb_0
        __hcse_4 = v[i_x]
        ubd[i_x] = ubd[i_x] + (__oldb_0 * vd[i_x] + __hcse_4 * __oldb_0d)
        ub[i_x] = ub[i_x] + __hcse_4 * __oldb_0
    end
    for i_x = i_n:-1:1
        __oldb_0d = vbd[i_x]
        __oldb_0 = vb[i_x]
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
        __hcse_5 = 2 * u[i_x]
        ubd[i_x] = ubd[i_x] + (__oldb_0 * (2 * ud[i_x]) + __hcse_5 * __oldb_0d)
        ub[i_x] = ub[i_x] + __hcse_5 * __oldb_0
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
