function ii_readnested_d(x, xd, y, yd, i_n, i_m, out, outd, acc, accd)
    vd = 0.0
    v = 1.0
    for i_i = 1:i_n
        vd = x[i_i] * xd[i_i] + x[i_i] * xd[i_i]
        v = x[i_i] * x[i_i]
        accd[i_i] = accd[i_i] + (v * vd + v * vd)
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            vd = y[i_j] * xd[i_i] + x[i_i] * yd[i_j]
            v = x[i_i] * y[i_j]
            outd[i_i] = outd[i_i] + (v * vd + v * vd)
            out[i_i] = out[i_i] + v * v
        end
    end
    return nothing
end

function ii_readnested(x, y, i_n, i_m, out, acc)
    v = 1.0
    for i_i = 1:i_n
        v = x[i_i] * x[i_i]
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            v = x[i_i] * y[i_j]
            out[i_i] = out[i_i] + v * v
        end
    end
end
