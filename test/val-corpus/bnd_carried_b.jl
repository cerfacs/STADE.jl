function initstacks_bnd_carried_b(i_n)
    t_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return t_stack
end

function bnd_carried_b(x, xb, y, yb, i_n, out, outb, t_stack)
    t = 0.0
    tb = 0.0
    t = 1.0
    for i_i = 1:i_n
        t = x[i_i] * y[i_i]
        out[i_i] = out[i_i] + t * t
        tb = tb + t * outb[i_i]
        tb = tb + t * outb[i_i]
        xb[i_i] = xb[i_i] + y[i_i] * tb
        yb[i_i] = yb[i_i] + x[i_i] * tb
        tb = 0.0
    end
    t_stack[max(0, div(i_n - 1, 1) + 1) + 1] = t
    t = t_stack[max(0, div(i_n - 1, 1) + 1) + 1]
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
