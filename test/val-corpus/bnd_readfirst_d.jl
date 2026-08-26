function bnd_readfirst_d(x, xd, i_n, i_m, out, outd)
    sd = 0.0
    s = 1.0
    for i_i = 1:i_n
        outd[i_i] = outd[i_i] + (s * sd + s * sd)
        out[i_i] = out[i_i] + s * s
        sd = 0.0
        s = 0.0
        for i_j = 1:i_m
            sd = sd + (x[i_j] * xd[i_i] + x[i_i] * xd[i_j])
            s = s + x[i_i] * x[i_j]
        end
    end
    return nothing
end

function bnd_readfirst(x, i_n, i_m, out)
    s = 1.0
    for i_i = 1:i_n
        out[i_i] = out[i_i] + s * s
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_i] * x[i_j]
        end
    end
end
