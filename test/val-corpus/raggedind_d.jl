function raggedind_d(out, outd, u, ud, x, xd, n, m0)
    for i_x = 1:n
        wd = 0.0
        w = m0 + i_x
        sd = 0.0
        s = 0.0
        for i_j = 1:w
            __cse_0 = u[i_x]
            __cse_1 = x[i_j]
            sd = sd + (__cse_0 * xd[i_j] + __cse_1 * ud[i_x])
            s = s + __cse_1 * __cse_0
        end
        __cse_2 = s * sd
        outd[i_x] = __cse_2 + __cse_2
        out[i_x] = s * s
    end
    return nothing
end

function raggedind(out, u, x, n, m0)
    for i_x = 1:n
        w = m0 + i_x
        s = 0.0
        for i_j = 1:w
            s = s + x[i_j] * u[i_x]
        end
        out[i_x] = s * s
    end
end
