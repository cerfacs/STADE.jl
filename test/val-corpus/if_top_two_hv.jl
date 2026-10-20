function initstacks_if_top_two_b()
    branch_stack = Vector{Int64}(undef, 1 + 1)
    s_stack = Vector{Float64}(undef, 1)
    t_stack = Vector{Float64}(undef, 1)
    return (branch_stack, s_stack, t_stack)
end

function if_top_two_hv(x, xb, y, yb, i_n, xd, xbd, yd, ybd, branch_stack, s_stack, t_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    t_stack_d = Vector{Float64}(undef, length(t_stack))
    s = 0.0
    t = 0.0
    sb = 0.0
    tb = 0.0
    sd = 0.0
    sbd = 0.0
    td = 0.0
    tbd = 0.0
    sd = 0.0
    s = 0.0
    if i_n > 2
        branch_stack[1] = 1
        __hcse_0 = x[2]
        __hcse_1 = x[1]
        sd = __hcse_0 * xd[1] + __hcse_1 * xd[2]
        s = __hcse_1 * __hcse_0
    else
        branch_stack[1] = 0
    end
    td = 0.0
    t = 0.0
    if i_n > 3
        __idx_branch_stack_0 = 1 + 1
        branch_stack[__idx_branch_stack_0] = 1
        __hcse_2 = x[3]
        __hcse_3 = x[2]
        td = __hcse_2 * xd[2] + __hcse_3 * xd[3]
        t = __hcse_3 * __hcse_2
    else
        __idx_branch_stack_0 = 1 + 1
        branch_stack[__idx_branch_stack_0] = 0
    end
    for i = 1:i_n
        __hcse_4 = x[i]
        yd[i] = yd[i] + (((t * __hcse_4) * sd + (s * __hcse_4) * td) + (s * t) * xd[i])
        y[i] = y[i] + s * t * __hcse_4
    end
    s_stack_d[1] = sd
    s_stack[1] = s
    t_stack_d[1] = td
    t_stack[1] = t
    sd = s_stack_d[1]
    s = s_stack[1]
    td = t_stack_d[1]
    t = t_stack[1]
    for i = i_n:-1:1
        __cse_0d = xd[i]
        __cse_0 = x[i]
        __cse_1d = ybd[i]
        __cse_1 = yb[i]
        __hcse_5 = t * __cse_0
        sbd = sbd + (__cse_1 * (__cse_0 * td + t * __cse_0d) + __hcse_5 * __cse_1d)
        sb = sb + __hcse_5 * __cse_1
        __hcse_6 = s * __cse_0
        tbd = tbd + (__cse_1 * (__cse_0 * sd + s * __cse_0d) + __hcse_6 * __cse_1d)
        tb = tb + __hcse_6 * __cse_1
        __hcse_7 = s * t
        xbd[i] = xbd[i] + (__cse_1 * (t * sd + s * td) + __hcse_7 * __cse_1d)
        xb[i] = xb[i] + __hcse_7 * __cse_1
    end
    __idx_branch_stack_3 = 1 + 1
    __branch = branch_stack[__idx_branch_stack_3]
    if __branch == 1
        __oldb_0d = tbd
        __oldb_0 = tb
        tbd = 0.0
        tb = 0.0
        __hcse_8 = x[3]
        xbd[2] = xbd[2] + (__oldb_0 * xd[3] + __hcse_8 * __oldb_0d)
        xb[2] = xb[2] + __hcse_8 * __oldb_0
        __hcse_9 = x[2]
        xbd[3] = xbd[3] + (__oldb_0 * xd[2] + __hcse_9 * __oldb_0d)
        xb[3] = xb[3] + __hcse_9 * __oldb_0
    end
    tbd = 0.0
    tb = 0.0
    __branch = branch_stack[1]
    if __branch == 1
        __oldb_0d = sbd
        __oldb_0 = sb
        sbd = 0.0
        sb = 0.0
        __hcse_10 = x[2]
        xbd[1] = xbd[1] + (__oldb_0 * xd[2] + __hcse_10 * __oldb_0d)
        xb[1] = xb[1] + __hcse_10 * __oldb_0
        __hcse_11 = x[1]
        xbd[2] = xbd[2] + (__oldb_0 * xd[1] + __hcse_11 * __oldb_0d)
        xb[2] = xb[2] + __hcse_11 * __oldb_0
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function if_top_two(x, y, i_n)
    s = 0.0
    if i_n > 2
        s = x[1] * x[2]
    end
    t = 0.0
    if i_n > 3
        t = x[2] * x[3]
    end
    for i = 1:i_n
        y[i] = y[i] + s * t * x[i]
    end
end
