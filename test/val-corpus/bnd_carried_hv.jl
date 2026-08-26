function initstacks_bnd_carried_b(i_n)
    t_stack = Vector{Float64}(undef, (div(i_n - 1, 1) + 1) + 1)
    return t_stack
end

function bnd_carried_hv(x, xb, y, yb, i_n, out, outb, xd, xbd, yd, ybd, outd, outbd, t_stack)
    t_stack_d = Vector{Float64}(undef, length(t_stack))
    t = 0.0
    tb = 0.0
    td = 0.0
    tbd = 0.0
    td = 0.0
    t = 1.0
    for i_i = 1:i_n
        td = y[i_i] * xd[i_i] + x[i_i] * yd[i_i]
        t = x[i_i] * y[i_i]
        outd[i_i] = outd[i_i] + (t * td + t * td)
        out[i_i] = out[i_i] + t * t
        tbd = tbd + (outb[i_i] * td + t * outbd[i_i])
        tb = tb + t * outb[i_i]
        tbd = tbd + (outb[i_i] * td + t * outbd[i_i])
        tb = tb + t * outb[i_i]
        xbd[i_i] = xbd[i_i] + (tb * yd[i_i] + y[i_i] * tbd)
        xb[i_i] = xb[i_i] + y[i_i] * tb
        ybd[i_i] = ybd[i_i] + (tb * xd[i_i] + x[i_i] * tbd)
        yb[i_i] = yb[i_i] + x[i_i] * tb
        tbd = 0.0
        tb = 0.0
    end
    t_stack_d[(div(i_n - 1, 1) + 1) + 1] = td
    t_stack[(div(i_n - 1, 1) + 1) + 1] = t
    td = t_stack_d[(div(i_n - 1, 1) + 1) + 1]
    t = t_stack[(div(i_n - 1, 1) + 1) + 1]
    tbd = 0.0
    tb = 0.0
    return nothing
end

function bnd_carried(x, y, i_n, out)
    t = 1.0
    for i_i = 1:i_n
        t = x[i_i] * y[i_i]
        out[i_i] = out[i_i] + t * t
    end
end
