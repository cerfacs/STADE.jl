function ii_readbefore_d(x, xd, i_n, i_m, out, outd)
    sd = x[1] * xd[1] + x[1] * xd[1]
    s = x[1] * x[1]
    outd[1] = outd[1] + (s * sd + s * sd)
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        sd = 0.0
        s = 0.0
        for i_j = 1:i_m
            sd = sd + (x[i_i] * xd[i_j] + x[i_j] * xd[i_i])
            s = s + x[i_j] * x[i_i]
        end
        outd[i_i] = outd[i_i] + (s * sd + s * sd)
        out[i_i] = out[i_i] + s * s
    end
    return nothing
end

function ii_readbefore(x, i_n, i_m, out)
    s = x[1] * x[1]
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_j] * x[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
end
