function initstacks_bnd_carried_b(i_n)
    t_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return t_stack
end

function bnd_carried_b(x, xb, y, yb, i_n, out, outb, t_stack)
    t = 0.0
    tb = 0.0
    t = 1.0
    for i_i = 1:i_n
        __cse_0 = x[i_i]
        __cse_1 = y[i_i]
        t = __cse_0 * __cse_1
        out[i_i] = out[i_i] + t * t
        __cse_2 = t * outb[i_i]
        tb = tb + __cse_2
        tb = tb + __cse_2
        __oldb_0 = tb
        tb = 0.0
        xb[i_i] = xb[i_i] + __cse_1 * __oldb_0
        yb[i_i] = yb[i_i] + __cse_0 * __oldb_0
    end
    __icse_3 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_t_stack_2 = __icse_3
    t_stack[__idx_t_stack_2] = t
    __idx_t_stack_0 = __icse_3
    t = t_stack[__idx_t_stack_0]
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
