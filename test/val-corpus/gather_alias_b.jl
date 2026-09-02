function initstacks_gather_alias_b(i_n)
    y_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1))
    return y_stack
end

function gather_alias_b(y, yb, x, xb, i_idx, i_n, y_stack)
    for i_r = 1:i_n
        i_k = i_idx[i_r]
        __idx_y_stack_1 = (i_r - 1) + 1
        y_stack[__idx_y_stack_1] = y[i_r]
        y[i_r] = y[i_k] * x[i_r]
    end
    for i_r = i_n:-1:1
        i_k = i_idx[i_r]
        __idx_y_stack_0 = (i_r - 1) + 1
        y[i_r] = y_stack[__idx_y_stack_0]
        __oldb_2 = yb[i_r]
        yb[i_r] = 0.0
        yb[i_k] = yb[i_k] + x[i_r] * __oldb_2
        xb[i_r] = xb[i_r] + y[i_k] * __oldb_2
    end
    return nothing
end

function gather_alias(y, x, i_idx, i_n)
    for i_r = 1:i_n
        i_k = i_idx[i_r]
        y[i_r] = y[i_k] * x[i_r]
    end
end
