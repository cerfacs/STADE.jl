function entry_empty_d(x, xd, u, ud, i_npass, i_w0, out, outd)
    __cse_0 = x[1]
    __cse_1 = __cse_0 * xd[1]
    sd = __cse_1 + __cse_1
    s = __cse_0 * __cse_0
    wd = 0.0
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            __cse_2 = u[i_j]
            __cse_3 = x[i_j]
            sd = __cse_2 * xd[i_j] + __cse_3 * ud[i_j]
            s = __cse_3 * __cse_2
        end
        __cse_4 = s * sd
        outd[i_p] = outd[i_p] + (__cse_4 + __cse_4)
        out[i_p] = out[i_p] + s * s
        wd = 0.0
        w = w - 3
    end
    return nothing
end

function entry_empty(x, u, i_npass, i_w0, out)
    s = x[1] * x[1]
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            s = x[i_j] * u[i_j]
        end
        out[i_p] = out[i_p] + s * s
        w = w - 3
    end
end
