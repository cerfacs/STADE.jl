function initstacks_gather_alias_b(i_n)
    y_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1))
    return y_stack
end

function gather_alias_hv(y, yb, x, xb, i_idx, i_n, yd, ybd, xd, xbd, y_stack)
    y_stack_d = Vector{Float64}(undef, length(y_stack))
    for i_r = 1:i_n
        i_k = i_idx[i_r]
        __idx_y_stack_1 = (i_r - 1) + 1
        y_stack_d[__idx_y_stack_1] = yd[i_r]
        y_stack[__idx_y_stack_1] = y[i_r]
        __hcse_0 = x[i_r]
        __hcse_1 = y[i_k]
        yd[i_r] = __hcse_0 * yd[i_k] + __hcse_1 * xd[i_r]
        y[i_r] = __hcse_1 * __hcse_0
    end
    for i_r = i_n:-1:1
        i_k = i_idx[i_r]
        __idx_y_stack_0 = (i_r - 1) + 1
        yd[i_r] = y_stack_d[__idx_y_stack_0]
        y[i_r] = y_stack[__idx_y_stack_0]
        __oldb_2d = ybd[i_r]
        __oldb_2 = yb[i_r]
        ybd[i_r] = 0.0
        yb[i_r] = 0.0
        __hcse_2 = x[i_r]
        ybd[i_k] = ybd[i_k] + (__oldb_2 * xd[i_r] + __hcse_2 * __oldb_2d)
        yb[i_k] = yb[i_k] + __hcse_2 * __oldb_2
        __hcse_3 = y[i_k]
        xbd[i_r] = xbd[i_r] + (__oldb_2 * yd[i_k] + __hcse_3 * __oldb_2d)
        xb[i_r] = xb[i_r] + __hcse_3 * __oldb_2
    end
    return nothing
end

function gather_alias(y, x, i_idx, i_n)
    for i_r = 1:i_n
        i_k = i_idx[i_r]
        y[i_r] = y[i_k] * x[i_r]
    end
end
