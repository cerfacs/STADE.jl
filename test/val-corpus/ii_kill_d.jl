function ii_kill_d(x, xd, u, ud, i_n, i_w0, out, outd, acc, accd)
    vd = 0.0
    v = 0.0
    for i_i = 1:i_n
        __cse_0 = x[i_i]
        __cse_1 = __cse_0 * xd[i_i]
        vd = __cse_1 + __cse_1
        v = __cse_0 * __cse_0
        accd[i_i] = accd[i_i] + vd
        acc[i_i] = acc[i_i] + v
    end
    wd = 0.0
    w = i_w0 - 5
    for i_j = 1:w
        __cse_2 = u[i_j]
        __cse_3 = __cse_2 * ud[i_j]
        vd = __cse_3 + __cse_3
        v = __cse_2 * __cse_2
    end
    __cse_4 = v * vd
    outd[1] = outd[1] + (__cse_4 + __cse_4)
    out[1] = out[1] + v * v
    return nothing
end

function ii_kill(x, u, i_n, i_w0, out, acc)
    v = 0.0
    for i_i = 1:i_n
        v = x[i_i] * x[i_i]
        acc[i_i] = acc[i_i] + v
    end
    w = i_w0 - 5
    for i_j = 1:w
        v = u[i_j] * u[i_j]
    end
    out[1] = out[1] + v * v
end
