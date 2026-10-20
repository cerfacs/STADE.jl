function initstacks_idm_src_consume_b(i_n)
    s_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    r_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return (s_stack, r_stack)
end

function idm_src_consume_b(u, ub, y, yb, i_n, s_stack, r_stack)
    r = 0.0
    s = 0.0
    rb = 0.0
    sb = 0.0
    for i_t = 1:i_n
        __icse_0 = i_t - 1
        i_b = 2__icse_0
        __icse_1 = __icse_0 + 1
        __idx_s_stack_1 = __icse_1
        s_stack[__idx_s_stack_1] = s
        s = u[i_b + 1]
        u[i_b + 1] = 0.0
        __idx_r_stack_5 = __icse_1
        r_stack[__idx_r_stack_5] = r
        r = u[i_b + 2]
        u[i_b + 2] = 0.0
        y[i_t] = s * r
    end
    __icse_2 = max(0, div(i_n - 1, 1) + 1) + 1
    __idx_r_stack_1 = __icse_2
    r_stack[__idx_r_stack_1] = r
    __idx_s_stack_3 = __icse_2
    s_stack[__idx_s_stack_3] = s
    __idx_r_stack_0 = __icse_2
    r = r_stack[__idx_r_stack_0]
    __idx_s_stack_2 = __icse_2
    s = s_stack[__idx_s_stack_2]
    for i_t = i_n:-1:1
        __icse_3 = i_t - 1
        i_b = 2__icse_3
        __oldb_0 = yb[i_t]
        yb[i_t] = 0.0
        sb = sb + r * __oldb_0
        rb = rb + s * __oldb_0
        ub[i_b + 2] = 0.0
        __icse_4 = __icse_3 + 1
        __idx_r_stack_0 = __icse_4
        r = r_stack[__idx_r_stack_0]
        __oldb_2 = rb
        rb = 0.0
        ub[i_b + 2] = ub[i_b + 2] + __oldb_2
        ub[i_b + 1] = 0.0
        __idx_s_stack_0 = __icse_4
        s = s_stack[__idx_s_stack_0]
        __oldb_2 = sb
        sb = 0.0
        ub[i_b + 1] = ub[i_b + 1] + __oldb_2
    end
    return nothing
end

function idm_src_consume(u, y, i_n)
    for i_t = 1:i_n
        i_b = 2 * (i_t - 1)
        s = u[i_b + 1]
        u[i_b + 1] = 0.0
        r = u[i_b + 2]
        u[i_b + 2] = 0.0
        y[i_t] = s * r
    end
end
