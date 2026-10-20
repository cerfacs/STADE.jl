function ddp_twoparam_d(loss, lossd, u, ud, w, wd, b, bd, h, hd, o, od, i_n, i_m)
    for i_j = 1:i_m
        sd = bd[i_j]
        s = b[i_j]
        for i_k = 1:i_n
            __cse_0 = u[i_k]
            __cse_1 = w[(i_j - 1) * i_n + i_k]
            sd = sd + (__cse_0 * wd[(i_j - 1) * i_n + i_k] + __cse_1 * ud[i_k])
            s = s + __cse_1 * __cse_0
        end
        hd[i_j] = sd
        h[i_j] = s
    end
    for i_j2 = 1:i_m
        od[i_j2] = hd[i_j2] + -(ud[1])
        o[i_j2] = h[i_j2] - u[1]
    end
    for i_j3 = 1:i_m
        __cse_2 = o[i_j3]
        lossd[1] = lossd[1] + (2__cse_2) * od[i_j3]
        loss[1] = loss[1] + __cse_2 ^ 2
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
