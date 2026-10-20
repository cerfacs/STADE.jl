function initstacks_if_top_two_b()
    branch_stack = Vector{Int64}(undef, 1 + 1)
    s_stack = Vector{Float64}(undef, 1)
    t_stack = Vector{Float64}(undef, 1)
    return (branch_stack, s_stack, t_stack)
end

function if_top_two_b(x, xb, y, yb, i_n, branch_stack, s_stack, t_stack)
    s = 0.0
    t = 0.0
    sb = 0.0
    tb = 0.0
    s = 0.0
    if i_n > 2
        branch_stack[1] = 1
        s = x[1] * x[2]
    else
        branch_stack[1] = 0
    end
    t = 0.0
    if i_n > 3
        __idx_branch_stack_0 = 1 + 1
        branch_stack[__idx_branch_stack_0] = 1
        t = x[2] * x[3]
    else
        __idx_branch_stack_0 = 1 + 1
        branch_stack[__idx_branch_stack_0] = 0
    end
    for i = 1:i_n
        y[i] = y[i] + s * t * x[i]
    end
    s_stack[1] = s
    t_stack[1] = t
    s = s_stack[1]
    t = t_stack[1]
    for i = i_n:-1:1
        __cse_0 = x[i]
        __cse_1 = yb[i]
        sb = sb + (t * __cse_0) * __cse_1
        tb = tb + (s * __cse_0) * __cse_1
        xb[i] = xb[i] + (s * t) * __cse_1
    end
    __idx_branch_stack_3 = 1 + 1
    __branch = branch_stack[__idx_branch_stack_3]
    if __branch == 1
        __oldb_0 = tb
        tb = 0.0
        xb[2] = xb[2] + x[3] * __oldb_0
        xb[3] = xb[3] + x[2] * __oldb_0
    end
    tb = 0.0
    __branch = branch_stack[1]
    if __branch == 1
        __oldb_0 = sb
        sb = 0.0
        xb[1] = xb[1] + x[2] * __oldb_0
        xb[2] = xb[2] + x[1] * __oldb_0
    end
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
