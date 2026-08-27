function red_escape_d(x, xd, y, yd, i_n, i_m, out, outd, acc, accd)
    sd = 0.0
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            sd = sd + (y[i_i] * xd[i_j] + x[i_j] * yd[i_i])
            s = s + x[i_j] * y[i_i]
        end
        outd[i_i] = outd[i_i] + (s * sd + s * sd)
        out[i_i] = out[i_i] + s * s
    end
    sd = y[1] * xd[1] + x[1] * yd[1]
    s = x[1] * y[1]
    accd[1] = accd[1] + (s * sd + s * sd)
    acc[1] = acc[1] + s * s
    return nothing
end

function red_escape(x, y, i_n, i_m, out, acc)
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            s = s + x[i_j] * y[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
    s = x[1] * y[1]
    acc[1] = acc[1] + s * s
end
