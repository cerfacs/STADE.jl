function bnd_nested_only_d(x, xd, y, yd, i_n, i_m, out, outd, acc, accd)
    vd = 0.0
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            vd = y[i_i] * xd[i_j] + x[i_j] * yd[i_i]
            v = x[i_j] * y[i_i]
            outd[i_i] = outd[i_i] + (v * vd + v * vd)
            out[i_i] = out[i_i] + v * v
        end
        outd[i_i] = outd[i_i] + (y[i_i] * outd[i_i - 1] + out[i_i - 1] * yd[i_i])
        out[i_i] = out[i_i] + out[i_i - 1] * y[i_i]
    end
    vd = y[1] * xd[1] + x[1] * yd[1]
    v = x[1] * y[1]
    accd[1] = accd[1] + (v * vd + v * vd)
    acc[1] = acc[1] + v * v
    return nothing
end

function bnd_nested_only(x, y, i_n, i_m, out, acc)
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            v = x[i_j] * y[i_i]
            out[i_i] = out[i_i] + v * v
        end
        out[i_i] = out[i_i] + out[i_i - 1] * y[i_i]
    end
    v = x[1] * y[1]
    acc[1] = acc[1] + v * v
end
