function initstacks_ddp_twoparam_b()
    return nothing
end

function ddp_twoparam_hv(loss, lossb, u, ub, w, wb, b, bb, h, hb, o, ob, i_n, i_m, lossd, lossbd, ud, ubd, wd, wbd, bd, bbd, hd, hbd, od, obd)
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_j = 1:i_m
        sd = bd[i_j]
        s = b[i_j]
        for i_k = 1:i_n
            __hcse_0 = u[i_k]
            __hcse_1 = w[(i_j - 1) * i_n + i_k]
            sd = sd + (__hcse_0 * wd[(i_j - 1) * i_n + i_k] + __hcse_1 * ud[i_k])
            s = s + __hcse_1 * __hcse_0
        end
        hd[i_j] = sd
        h[i_j] = s
    end
    for i_j2 = 1:i_m
        od[i_j2] = hd[i_j2] + -(ud[1])
        o[i_j2] = h[i_j2] - u[1]
    end
    for i_j3 = 1:i_m
        __hcse_2 = o[i_j3]
        lossd[1] = lossd[1] + (2__hcse_2) * od[i_j3]
        loss[1] = loss[1] + __hcse_2 ^ 2
    end
    for i_j3 = i_m:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = 2 * o[i_j3]
        obd[i_j3] = obd[i_j3] + (__hcse_3 * (2 * od[i_j3]) + __hcse_4 * lossbd[1])
        ob[i_j3] = ob[i_j3] + __hcse_4 * __hcse_3
    end
    for i_j2 = i_m:-1:1
        __oldb_0d = obd[i_j2]
        __oldb_0 = ob[i_j2]
        obd[i_j2] = 0.0
        ob[i_j2] = 0.0
        hbd[i_j2] = hbd[i_j2] + __oldb_0d
        hb[i_j2] = hb[i_j2] + __oldb_0
        ubd[1] = ubd[1] + -__oldb_0d
        ub[1] = ub[1] + -__oldb_0
    end
    for i_j = i_m:-1:1
        __oldb_0d = hbd[i_j]
        __oldb_0 = hb[i_j]
        hbd[i_j] = 0.0
        hb[i_j] = 0.0
        sbd = sbd + __oldb_0d
        sb = sb + __oldb_0
        for i_k = i_n:-1:1
            __hcse_5 = u[i_k]
            wbd[(i_j - 1) * i_n + i_k] = wbd[(i_j - 1) * i_n + i_k] + (sb * ud[i_k] + __hcse_5 * sbd)
            wb[(i_j - 1) * i_n + i_k] = wb[(i_j - 1) * i_n + i_k] + __hcse_5 * sb
            __hcse_6 = w[(i_j - 1) * i_n + i_k]
            ubd[i_k] = ubd[i_k] + (sb * wd[(i_j - 1) * i_n + i_k] + __hcse_6 * sbd)
            ub[i_k] = ub[i_k] + __hcse_6 * sb
        end
        __oldb_0d = sbd
        __oldb_0 = sb
        sbd = 0.0
        sb = 0.0
        bbd[i_j] = bbd[i_j] + __oldb_0d
        bb[i_j] = bb[i_j] + __oldb_0
    end
    return nothing
end

function ddp_twoparam(loss, u, w, b, h, o, i_n, i_m)
    for i_j = 1:i_m
        s = b[i_j]
        for i_k = 1:i_n
            s = s + w[(i_j - 1) * i_n + i_k] * u[i_k]
        end
        h[i_j] = s
    end
    for i_j2 = 1:i_m
        o[i_j2] = h[i_j2] - u[1]
    end
    for i_j3 = 1:i_m
        loss[1] = loss[1] + o[i_j3] ^ 2
    end
end
