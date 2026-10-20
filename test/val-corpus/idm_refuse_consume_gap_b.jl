function initstacks_idm_refuse_consume_gap_b(i_n)
    s_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return s_stack
end

function idm_refuse_consume_gap_b(u, ub, x, xb, i_n, s_stack)
    s = 0.0
    sb = 0.0
    for i_t = 1:i_n
        __idx_s_stack_0 = (i_t - 1) + 1
        s_stack[__idx_s_stack_0] = s
        s = u[i_t]
        u[i_t + 1] = s * x[i_t] + 1.0
        u[i_t] = 0.0
    end
    __icse_0 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_s_stack_1 = __icse_0
    s_stack[__idx_s_stack_1] = s
    __idx_s_stack_0 = __icse_0
    s = s_stack[__idx_s_stack_0]
    for i_t = i_n:-1:1
        ub[i_t] = 0.0
        __oldb_0 = ub[i_t + 1]
        ub[i_t + 1] = 0.0
        sb = sb + x[i_t] * __oldb_0
        xb[i_t] = xb[i_t] + s * __oldb_0
        __idx_s_stack_0 = (i_t - 1) + 1
        s = s_stack[__idx_s_stack_0]
        __oldb_2 = sb
        sb = 0.0
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
