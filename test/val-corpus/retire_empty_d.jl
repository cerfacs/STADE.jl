function retire_empty_d(u, ud, x, xd, i_npass, i_w0)
    wd = 0.0
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            __cse_0 = ud[i_j]
            __cse_1 = u[i_j]
            __cse_2 = x[i_j]
            ud[i_j] = __cse_0 + (__cse_1 * xd[i_j] + __cse_2 * __cse_0)
            u[i_j] = __cse_1 + __cse_2 * __cse_1
        end
        wd = 0.0
        w = w - 3
    end
    return nothing
end

function retire_empty(u, x, i_npass, i_w0)
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            u[i_j] = u[i_j] + x[i_j] * u[i_j]
        end
        w = w - 3
    end
end
