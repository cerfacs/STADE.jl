function red_nested_inner_d(out, outd, u, ud, v, vd, i_n, i_m)
    for i_j = 1:i_m
        sd = 0.0
        s = 0.0
        for i_k = 1:i_n
            __cse_0 = u[i_k]
            __cse_1 = v[(i_j - 1) * i_n + i_k]
            sd = sd + (__cse_0 * vd[(i_j - 1) * i_n + i_k] + __cse_1 * ud[i_k])
            s = s + __cse_1 * __cse_0
        end
        __cse_2 = s * sd
        outd[i_j] = __cse_2 + __cse_2
        out[i_j] = s * s
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
