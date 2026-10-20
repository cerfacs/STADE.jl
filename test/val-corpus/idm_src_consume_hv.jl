function initstacks_idm_src_consume_b(i_n)
    s_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    r_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + 1)
    return (s_stack, r_stack)
end

function idm_src_consume_hv(u, ub, y, yb, i_n, ud, ubd, yd, ybd, s_stack, r_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    r_stack_d = Vector{Float64}(undef, length(r_stack))
    r = 0.0
    s = 0.0
    rb = 0.0
    sb = 0.0
    rd = 0.0
    rbd = 0.0
    sd = 0.0
    sbd = 0.0
    for i_t = 1:i_n
        __icse_0 = i_t - 1
        i_b = 2__icse_0
        __icse_1 = __icse_0 + 1
        __idx_s_stack_1 = __icse_1
        s_stack_d[__idx_s_stack_1] = sd
        s_stack[__idx_s_stack_1] = s
        sd = ud[i_b + 1]
        s = u[i_b + 1]
        ud[i_b + 1] = 0.0
        u[i_b + 1] = 0.0
        __idx_r_stack_5 = __icse_1
        r_stack_d[__idx_r_stack_5] = rd
        r_stack[__idx_r_stack_5] = r
        rd = ud[i_b + 2]
        r = u[i_b + 2]
        ud[i_b + 2] = 0.0
        u[i_b + 2] = 0.0
        yd[i_t] = r * sd + s * rd
        y[i_t] = s * r
    end
    __ihcse_0 = max(0, div(i_n - 1, 1) + 1) + 1
    __icse_2 = __ihcse_0
    __idx_r_stack_1 = __icse_2
    r_stack_d[__idx_r_stack_1] = rd
    r_stack[__idx_r_stack_1] = r
    __idx_s_stack_3 = __icse_2
    s_stack_d[__idx_s_stack_3] = sd
    s_stack[__idx_s_stack_3] = s
    __icse_3 = __ihcse_0
    __idx_r_stack_0 = __icse_3
    rd = r_stack_d[__idx_r_stack_0]
    r = r_stack[__idx_r_stack_0]
    __idx_s_stack_2 = __icse_3
    sd = s_stack_d[__idx_s_stack_2]
    s = s_stack[__idx_s_stack_2]
    for i_t = i_n:-1:1
        __icse_4 = i_t - 1
        i_b = 2__icse_4
        __oldb_0d = ybd[i_t]
        __oldb_0 = yb[i_t]
        ybd[i_t] = 0.0
        yb[i_t] = 0.0
        sbd = sbd + (__oldb_0 * rd + r * __oldb_0d)
        sb = sb + r * __oldb_0
        rbd = rbd + (__oldb_0 * sd + s * __oldb_0d)
        rb = rb + s * __oldb_0
        ubd[i_b + 2] = 0.0
        ub[i_b + 2] = 0.0
        __icse_5 = __icse_4 + 1
        __idx_r_stack_0 = __icse_5
        rd = r_stack_d[__idx_r_stack_0]
        r = r_stack[__idx_r_stack_0]
        __oldb_2d = rbd
        __oldb_2 = rb
        rbd = 0.0
        rb = 0.0
        ubd[i_b + 2] = ubd[i_b + 2] + __oldb_2d
        ub[i_b + 2] = ub[i_b + 2] + __oldb_2
        ubd[i_b + 1] = 0.0
        ub[i_b + 1] = 0.0
        __idx_s_stack_0 = __icse_5
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __oldb_2d = sbd
        __oldb_2 = sb
        sbd = 0.0
        sb = 0.0
        ubd[i_b + 1] = ubd[i_b + 1] + __oldb_2d
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
