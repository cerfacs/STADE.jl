function initstacks_red_nested_inner_b(i_m)
    return nothing
end

function red_nested_inner_hv(out, outb, u, ub, v, vb, i_n, i_m, outd, outbd, ud, ubd, vd, vbd)
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_j = 1:i_m
        sd = 0.0
        s = 0.0
        for i_k = 1:i_n
            __hcse_0 = u[i_k]
            __hcse_1 = v[(i_j - 1) * i_n + i_k]
            sd = sd + (__hcse_0 * vd[(i_j - 1) * i_n + i_k] + __hcse_1 * ud[i_k])
            s = s + __hcse_1 * __hcse_0
        end
        __hcse_2 = s * sd
        outd[i_j] = __hcse_2 + __hcse_2
        out[i_j] = s * s
        __oldb_0d = outbd[i_j]
        __oldb_0 = outb[i_j]
        outbd[i_j] = 0.0
        outb[i_j] = 0.0
        __cse_0d = __oldb_0 * sd + s * __oldb_0d
        __cse_0 = s * __oldb_0
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        for i_k = 1:i_n
            __cse_1d = vd[(i_j - 1) * i_n + i_k]
            __cse_1 = v[(i_j - 1) * i_n + i_k]
            __cse_2d = ud[i_k]
            __cse_2 = u[i_k]
            sd = sd + (__cse_2 * __cse_1d + __cse_1 * __cse_2d)
            s = s + __cse_1 * __cse_2
            vbd[(i_j - 1) * i_n + i_k] = vbd[(i_j - 1) * i_n + i_k] + (sb * __cse_2d + __cse_2 * sbd)
            vb[(i_j - 1) * i_n + i_k] = vb[(i_j - 1) * i_n + i_k] + __cse_2 * sb
            ubd[i_k] = ubd[i_k] + (sb * __cse_1d + __cse_1 * sbd)
            ub[i_k] = ub[i_k] + __cse_1 * sb
        end
        sbd = 0.0
        sb = 0.0
    end
    return nothing
end

function red_nested_inner(out, u, v, i_n, i_m)
    for i_j = 1:i_m
        s = 0.0
        for i_k = 1:i_n
            s = s + v[(i_j - 1) * i_n + i_k] * u[i_k]
        end
        out[i_j] = s * s
    end
end
