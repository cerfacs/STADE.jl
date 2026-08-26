function bnd_carried_d(x, xd, y, yd, i_n, out, outd)
    td = 0.0
    t = 1.0
    for i_i = 1:i_n
        td = y[i_i] * xd[i_i] + x[i_i] * yd[i_i]
        t = x[i_i] * y[i_i]
        outd[i_i] = outd[i_i] + (t * td + t * td)
        out[i_i] = out[i_i] + t * t
    end
    return nothing
end

function bnd_carried(x, y, i_n, out)
    t = 1.0
    for i_i = 1:i_n
        t = x[i_i] * y[i_i]
        out[i_i] = out[i_i] + t * t
    end
end
