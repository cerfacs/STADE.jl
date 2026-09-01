function bnd_readfirst_d(x, xd, i_n, i_m, out, outd)
    sd = 0.0
    s = 1.0
    for i_i = 1:i_n
        __cse_0 = s * sd
        outd[i_i] = outd[i_i] + (__cse_0 + __cse_0)
        out[i_i] = out[i_i] + s * s
        sd = 0.0
        s = 0.0
        for i_j = 1:i_m
            __cse_1 = x[i_j]
            __cse_2 = x[i_i]
            sd = sd + (__cse_1 * xd[i_i] + __cse_2 * xd[i_j])
            s = s + __cse_2 * __cse_1
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
