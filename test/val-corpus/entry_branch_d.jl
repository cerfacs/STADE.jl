function entry_branch_d(x, xd, y, yd, flag, flagd, i_n, out, outd)
    __cse_0 = y[1]
    __cse_1 = x[1]
    sd = __cse_0 * xd[1] + __cse_1 * yd[1]
    s = __cse_1 * __cse_0
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            __cse_2 = y[i_i]
            __cse_3 = x[i_i]
            sd = __cse_2 * xd[i_i] + __cse_3 * yd[i_i]
            s = __cse_3 * __cse_2
        end
        __cse_4 = s * sd
        outd[i_i] = outd[i_i] + (__cse_4 + __cse_4)
        out[i_i] = out[i_i] + s * s
    end
    return nothing
end

function entry_branch(x, y, flag, i_n, out)
    s = x[1] * y[1]
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            s = x[i_i] * y[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
end
