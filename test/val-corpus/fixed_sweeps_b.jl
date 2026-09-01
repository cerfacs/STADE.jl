function initstacks_fixed_sweeps_b(i_n)
    y_stack = Vector{Float64}(undef, max(0, div(3 - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    return y_stack
end

function fixed_sweeps_b(y, yb, x, xb, i_n, y_stack)
    for i_k = 1:3
        for i_i = 1:i_n
            __idx_y_stack_0 = ((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_i - 1)) + 1
            __cse_0 = y[i_i]
            y_stack[__idx_y_stack_0] = __cse_0
            y[i_i] = 0.5__cse_0 + x[i_i]
        end
    end
    for i_k = 3:-1:1
        for i_i = i_n:-1:1
            __idx_y_stack_0 = ((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_i - 1)) + 1
            y[i_i] = y_stack[__idx_y_stack_0]
            __cse_1 = yb[i_i]
            xb[i_i] = xb[i_i] + __cse_1
            yb[i_i] = 0.5__cse_1
        end
    end
    return nothing
end

function fixed_sweeps(y, x, i_n)
    for i_k = 1:3
        for i_i = 1:i_n
            y[i_i] = 0.5 * y[i_i] + x[i_i]
        end
    end
end
