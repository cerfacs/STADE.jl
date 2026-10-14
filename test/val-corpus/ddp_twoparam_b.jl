function initstacks_ddp_twoparam_b()
    return nothing
end

function ddp_twoparam_b(loss, lossb, u, ub, w, wb, b, bb, h, hb, o, ob, i_n, i_m)
    s = 0.0
    sb = 0.0
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
    for i_j3 = i_m:-1:1
        ob[i_j3] = ob[i_j3] + (2 * o[i_j3]) * lossb[1]
    end
    for i_j2 = i_m:-1:1
        __oldb_0 = ob[i_j2]
        ob[i_j2] = 0.0
        hb[i_j2] = hb[i_j2] + __oldb_0
        ub[1] = ub[1] + -__oldb_0
    end
    for i_j = i_m:-1:1
        __oldb_0 = hb[i_j]
        hb[i_j] = 0.0
        sb = sb + __oldb_0
        for i_k = i_n:-1:1
            wb[(i_j - 1) * i_n + i_k] = wb[(i_j - 1) * i_n + i_k] + u[i_k] * sb
            ub[i_k] = ub[i_k] + w[(i_j - 1) * i_n + i_k] * sb
        end
        __oldb_0 = sb
        sb = 0.0
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
