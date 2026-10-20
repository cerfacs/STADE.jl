function initstacks_if_top_scalar_b()
    branch_stack = Vector{Int64}(undef, 1)
    s_stack = Vector{Float64}(undef, 1)
    return (branch_stack, s_stack)
end

function if_top_scalar_hv(x, xb, y, yb, i_n, xd, xbd, yd, ybd, branch_stack, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    if x[1] > 0.0
        branch_stack[1] = 1
        __hcse_0 = x[2]
        __hcse_1 = x[1]
        sd = __hcse_0 * xd[1] + __hcse_1 * xd[2]
        s = __hcse_1 * __hcse_0
    else
        branch_stack[1] = 0
        __cse_0d = xd[2]
        __cse_0 = x[2]
        __hcse_2 = __cse_0 * __cse_0d
        sd = __hcse_2 + __hcse_2
        s = __cse_0 * __cse_0
    end
    for i = 1:i_n
        __hcse_3 = x[i]
        __hcse_4 = (s * __hcse_3) * sd
        yd[i] = yd[i] + ((__hcse_4 + __hcse_4) + (s * s) * xd[i])
        y[i] = y[i] + s * s * __hcse_3
    end
    s_stack_d[1] = sd
    s_stack[1] = s
    sd = s_stack_d[1]
    s = s_stack[1]
    for i = i_n:-1:1
        __cse_1d = ybd[i]
        __cse_1 = yb[i]
        __hcse_5 = x[i]
        __hcse_6 = s * __hcse_5
        __cse_2d = __cse_1 * (__hcse_5 * sd + s * xd[i]) + __hcse_6 * __cse_1d
        __cse_2 = __hcse_6 * __cse_1
        sbd = sbd + __cse_2d
        sb = sb + __cse_2
        sbd = sbd + __cse_2d
        sb = sb + __cse_2
        __hcse_7 = s * sd
        __hcse_8 = s * s
        xbd[i] = xbd[i] + (__cse_1 * (__hcse_7 + __hcse_7) + __hcse_8 * __cse_1d)
        xb[i] = xb[i] + __hcse_8 * __cse_1
    end
    __branch = branch_stack[1]
    if __branch == 1
        __oldb_0d = sbd
        __oldb_0 = sb
        sbd = 0.0
        sb = 0.0
        __hcse_9 = x[2]
        xbd[1] = xbd[1] + (__oldb_0 * xd[2] + __hcse_9 * __oldb_0d)
        xb[1] = xb[1] + __hcse_9 * __oldb_0
        __hcse_10 = x[1]
        xbd[2] = xbd[2] + (__oldb_0 * xd[1] + __hcse_10 * __oldb_0d)
        xb[2] = xb[2] + __hcse_10 * __oldb_0
    else
        __oldb_0d = sbd
        __oldb_0 = sb
        sbd = 0.0
        sb = 0.0
        __hcse_11 = x[2]
        __cse_3d = __oldb_0 * xd[2] + __hcse_11 * __oldb_0d
        __cse_3 = __hcse_11 * __oldb_0
        xbd[2] = xbd[2] + __cse_3d
        xb[2] = xb[2] + __cse_3
        xbd[2] = xbd[2] + __cse_3d
        xb[2] = xb[2] + __cse_3
    end
    return nothing
end

function if_top_scalar(x, y, i_n)
    if x[1] > 0.0
        s = x[1] * x[2]
    else
        s = x[2] * x[2]
    end
    for i = 1:i_n
        y[i] = y[i] + s * s * x[i]
    end
end
