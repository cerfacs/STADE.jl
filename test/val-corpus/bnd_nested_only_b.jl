function initstacks_bnd_nested_only_b(i_m, i_n)
    v_stack = Vector{Float64}(undef, ((max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1) + 1)
    out_stack = Vector{Float64}(undef, max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1))
    return (v_stack, out_stack)
end

function bnd_nested_only_b(x, xb, y, yb, i_n, i_m, out, outb, acc, accb, v_stack, out_stack)
    v = 0.0
    vb = 0.0
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            __idx_v_stack_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            v_stack[__idx_v_stack_0] = v
            v = x[i_j] * y[i_i]
            __idx_out_stack_3 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            __cse_0 = out[i_i]
            out_stack[__idx_out_stack_3] = __cse_0
            out[i_i] = __cse_0 + v * v
        end
        __idx_out_stack_1 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        __cse_1 = out[i_i]
        out_stack[__idx_out_stack_1] = __cse_1
        out[i_i] = __cse_1 + out[i_i - 1] * y[i_i]
        __idx_v_stack_4 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        v_stack[__idx_v_stack_4] = v
    end
    __idx_v_stack_2 = (max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1
    v_stack[__idx_v_stack_2] = v
    __cse_2 = x[1]
    __cse_3 = y[1]
    v = __cse_2 * __cse_3
    acc[1] = acc[1] + v * v
    __idx_v_stack_6 = ((max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1) + 1
    v_stack[__idx_v_stack_6] = v
    __idx_v_stack_0 = ((max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1) + 1
    v = v_stack[__idx_v_stack_0]
    __cse_4 = v * accb[1]
    vb = vb + __cse_4
    vb = vb + __cse_4
    __idx_v_stack_0 = (max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1
    v = v_stack[__idx_v_stack_0]
    xb[1] = xb[1] + __cse_3 * vb
    yb[1] = yb[1] + __cse_2 * vb
    vb = 0.0
    for i_i = i_n:-1:2
        __idx_v_stack_0 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        v = v_stack[__idx_v_stack_0]
        __idx_out_stack_0 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        out[i_i] = out_stack[__idx_out_stack_0]
        outb[i_i - 1] = outb[i_i - 1] + y[i_i] * outb[i_i]
        yb[i_i] = yb[i_i] + out[i_i - 1] * outb[i_i]
        for i_j = i_m:-1:1
            __idx_out_stack_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            out[i_i] = out_stack[__idx_out_stack_0]
            __cse_5 = v * outb[i_i]
            vb = vb + __cse_5
            vb = vb + __cse_5
            __idx_v_stack_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            v = v_stack[__idx_v_stack_0]
            xb[i_j] = xb[i_j] + y[i_i] * vb
            yb[i_i] = yb[i_i] + x[i_j] * vb
            vb = 0.0
        end
    end
    vb = 0.0
    return nothing
end

function bnd_nested_only(x, y, i_n, i_m, out, acc)
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            v = x[i_j] * y[i_i]
            out[i_i] = out[i_i] + v * v
        end
        out[i_i] = out[i_i] + out[i_i - 1] * y[i_i]
    end
    v = x[1] * y[1]
    acc[1] = acc[1] + v * v
end
