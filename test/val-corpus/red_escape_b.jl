function initstacks_red_escape_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, ((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1)
    return s_stack
end

function red_escape_b(x, xb, y, yb, i_n, i_m, out, outb, acc, accb, s_stack)
    s = 0.0
    sb = 0.0
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            __idx_s_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            s_stack[__idx_s_stack_0] = s
            s = s + x[i_j] * y[i_i]
        end
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_2 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        s_stack[__idx_s_stack_2] = s
    end
    __idx_s_stack_2 = (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1
    s_stack[__idx_s_stack_2] = s
    __cse_0 = x[1]
    __cse_1 = y[1]
    s = __cse_0 * __cse_1
    acc[1] = acc[1] + s * s
    __idx_s_stack_6 = ((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1
    s_stack[__idx_s_stack_6] = s
    __idx_s_stack_0 = ((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1
    s = s_stack[__idx_s_stack_0]
    __cse_2 = s * accb[1]
    sb = sb + __cse_2
    sb = sb + __cse_2
    __idx_s_stack_0 = (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1
    s = s_stack[__idx_s_stack_0]
    xb[1] = xb[1] + __cse_1 * sb
    yb[1] = yb[1] + __cse_0 * sb
    sb = 0.0
    for i_i = i_n:-1:1
        __idx_s_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        s = s_stack[__idx_s_stack_0]
        __cse_3 = s * outb[i_i]
        sb = sb + __cse_3
        sb = sb + __cse_3
        for i_j = i_m:-1:1
            __idx_s_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            s = s_stack[__idx_s_stack_0]
            xb[i_j] = xb[i_j] + y[i_i] * sb
            yb[i_i] = yb[i_i] + x[i_j] * sb
        end
    end
    sb = 0.0
    return nothing
end

function red_escape(x, y, i_n, i_m, out, acc)
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            s = s + x[i_j] * y[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
    s = x[1] * y[1]
    acc[1] = acc[1] + s * s
end
