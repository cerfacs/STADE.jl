function initstacks_idm_refuse_consume_gap_b(i_n)
    s_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return s_stack
end

function idm_refuse_consume_gap_hv(u, ub, x, xb, i_n, ud, ubd, xd, xbd, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_t = 1:i_n
        __idx_s_stack_0 = (i_t - 1) + 1
        s_stack_d[__idx_s_stack_0] = sd
        s_stack[__idx_s_stack_0] = s
        sd = ud[i_t]
        s = u[i_t]
        __hcse_0 = x[i_t]
        ud[i_t + 1] = __hcse_0 * sd + s * xd[i_t]
        u[i_t + 1] = s * __hcse_0 + 1.0
        ud[i_t] = 0.0
        u[i_t] = 0.0
    end
    __ihcse_1 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_s_stack_1 = __ihcse_1
    s_stack_d[__idx_s_stack_1] = sd
    s_stack[__idx_s_stack_1] = s
    __idx_s_stack_0 = __ihcse_1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_t = i_n:-1:1
        ubd[i_t] = 0.0
        ub[i_t] = 0.0
        __oldb_0d = ubd[i_t + 1]
        __oldb_0 = ub[i_t + 1]
        ubd[i_t + 1] = 0.0
        ub[i_t + 1] = 0.0
        __hcse_2 = x[i_t]
        sbd = sbd + (__oldb_0 * xd[i_t] + __hcse_2 * __oldb_0d)
        sb = sb + __hcse_2 * __oldb_0
        xbd[i_t] = xbd[i_t] + (__oldb_0 * sd + s * __oldb_0d)
        xb[i_t] = xb[i_t] + s * __oldb_0
        __idx_s_stack_0 = (i_t - 1) + 1
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __oldb_2d = sbd
        __oldb_2 = sb
        sbd = 0.0
        sb = 0.0
        ubd[i_t] = ubd[i_t] + __oldb_2d
        ub[i_t] = ub[i_t] + __oldb_2
    end
    return nothing
end

function idm_refuse_consume_gap(u, x, i_n)
    for i_t = 1:i_n
        s = u[i_t]
        u[i_t + 1] = s * x[i_t] + 1.0
        u[i_t] = 0.0
    end
end
