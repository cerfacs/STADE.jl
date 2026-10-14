function initstacks_red_nested_inner_b(i_m)
    return nothing
end

function red_nested_inner_b(out, outb, u, ub, v, vb, i_n, i_m)
    s = 0.0
    sb = 0.0
    for i_j = 1:i_m
        s = 0.0
        for i_k = 1:i_n
            s = s + v[(i_j - 1) * i_n + i_k] * u[i_k]
        end
        out[i_j] = s * s
        __oldb_0 = outb[i_j]
        outb[i_j] = 0.0
        __cse_0 = s * __oldb_0
        sb = sb + __cse_0
        sb = sb + __cse_0
        for i_k = 1:i_n
            __cse_1 = v[(i_j - 1) * i_n + i_k]
            __cse_2 = u[i_k]
            s = s + __cse_1 * __cse_2
            vb[(i_j - 1) * i_n + i_k] = vb[(i_j - 1) * i_n + i_k] + __cse_2 * sb
            ub[i_k] = ub[i_k] + __cse_1 * sb
        end
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
