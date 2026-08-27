function entry_branch_d(x, xd, y, yd, flag, flagd, i_n, out, outd)
    sd = y[1] * xd[1] + x[1] * yd[1]
    s = x[1] * y[1]
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            sd = y[i_i] * xd[i_i] + x[i_i] * yd[i_i]
            s = x[i_i] * y[i_i]
        end
        outd[i_i] = outd[i_i] + (s * sd + s * sd)
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
