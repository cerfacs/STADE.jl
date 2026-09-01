function initstacks_bnd_carried_b(i_n)
    t_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
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
        __cse_0d = xd[i_i]
        __cse_0 = x[i_i]
        __cse_1d = yd[i_i]
        __cse_1 = y[i_i]
        td = __cse_1 * __cse_0d + __cse_0 * __cse_1d
        t = __cse_0 * __cse_1
        __cse_3 = t * td
        outd[i_i] = outd[i_i] + (__cse_3 + __cse_3)
        out[i_i] = out[i_i] + t * t
        __cse_4 = outb[i_i]
        __cse_2d = __cse_4 * td + t * outbd[i_i]
        __cse_2 = t * __cse_4
        tbd = tbd + __cse_2d
        tb = tb + __cse_2
        tbd = tbd + __cse_2d
        tb = tb + __cse_2
        xbd[i_i] = xbd[i_i] + (tb * __cse_1d + __cse_1 * tbd)
        xb[i_i] = xb[i_i] + __cse_1 * tb
        ybd[i_i] = ybd[i_i] + (tb * __cse_0d + __cse_0 * tbd)
        yb[i_i] = yb[i_i] + __cse_0 * tb
        tbd = 0.0
        tb = 0.0
    end
    __idx_t_stack_2 = max(0, div(i_n - 1, 1) + 1) + 1
    t_stack_d[__idx_t_stack_2] = td
    t_stack[__idx_t_stack_2] = t
    __idx_t_stack_0 = max(0, div(i_n - 1, 1) + 1) + 1
    td = t_stack_d[__idx_t_stack_0]
    t = t_stack[__idx_t_stack_0]
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
