function coarsen_retire_d(x, xd, y, yd, n, levels, out, outd)
    curd = 0.0
    cur = n
    for i_l = 1:levels
        for i = 1:cur
            td = x[i] * xd[i] + x[i] * xd[i]
            t = x[i] * x[i]
            yd[i] = yd[i] + (t * td + t * td)
            y[i] = y[i] + t * t
        end
        curd = 0.0
        cur = div(cur + 1, 2)
    end
    outd[1] = yd[1]
    out[1] = y[1]
    return nothing
end

function coarsen_retire(x, y, n, levels, out)
    cur = n
    for i_l = 1:levels
        for i = 1:cur
            t = x[i] * x[i]
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
end
