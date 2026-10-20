function initstacks_if_top_scalar_b()
    branch_stack = Vector{Int64}(undef, 1)
    s_stack = Vector{Float64}(undef, 1)
    return (branch_stack, s_stack)
end

function if_top_scalar_b(x, xb, y, yb, i_n, branch_stack, s_stack)
    s = 0.0
    sb = 0.0
    if x[1] > 0.0
        branch_stack[1] = 1
        s = x[1] * x[2]
    else
        branch_stack[1] = 0
        __cse_0 = x[2]
        s = __cse_0 * __cse_0
    end
    for i = 1:i_n
        y[i] = y[i] + s * s * x[i]
    end
    s_stack[1] = s
    s = s_stack[1]
    for i = i_n:-1:1
        __cse_1 = yb[i]
        __cse_2 = (s * x[i]) * __cse_1
        sb = sb + __cse_2
        sb = sb + __cse_2
        xb[i] = xb[i] + (s * s) * __cse_1
    end
    __branch = branch_stack[1]
    if __branch == 1
        __oldb_0 = sb
        sb = 0.0
        xb[1] = xb[1] + x[2] * __oldb_0
        xb[2] = xb[2] + x[1] * __oldb_0
    else
        __oldb_0 = sb
        sb = 0.0
        __cse_3 = x[2] * __oldb_0
        xb[2] = xb[2] + __cse_3
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
