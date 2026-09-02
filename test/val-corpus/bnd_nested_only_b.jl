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
            __icse_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            __idx_v_stack_0 = __icse_0
            v_stack[__idx_v_stack_0] = v
            v = x[i_j] * y[i_i]
            __idx_out_stack_3 = __icse_0
            __cse_1 = out[i_i]
            out_stack[__idx_out_stack_3] = __cse_1
            out[i_i] = __cse_1 + v * v
        end
        __icse_2 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        __idx_out_stack_1 = __icse_2
        __cse_3 = out[i_i]
        out_stack[__idx_out_stack_1] = __cse_3
        out[i_i] = __cse_3 + out[i_i - 1] * y[i_i]
        __idx_v_stack_4 = __icse_2
        v_stack[__idx_v_stack_4] = v
    end
    __icse_4 = max(0, div(i_n - 2, 1) + 1)
    __icse_5 = (__icse_4 * max(0, div(i_m - 1, 1) + 1) + __icse_4) + 1
    __idx_v_stack_2 = __icse_5
    v_stack[__idx_v_stack_2] = v
    __cse_6 = x[1]
    __cse_7 = y[1]
    v = __cse_6 * __cse_7
    acc[1] = acc[1] + v * v
    __icse_8 = __icse_5 + 1
    __idx_v_stack_6 = __icse_8
    v_stack[__idx_v_stack_6] = v
    __idx_v_stack_0 = __icse_8
    v = v_stack[__idx_v_stack_0]
    __cse_9 = v * accb[1]
    vb = vb + __cse_9
    vb = vb + __cse_9
    __idx_v_stack_0 = __icse_5
    v = v_stack[__idx_v_stack_0]
    __oldb_2 = vb
    vb = 0.0
    xb[1] = xb[1] + __cse_7 * __oldb_2
    yb[1] = yb[1] + __cse_6 * __oldb_2
    for i_i = i_n:-1:2
        __icse_10 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        __idx_v_stack_0 = __icse_10
        v = v_stack[__idx_v_stack_0]
        __idx_out_stack_0 = __icse_10
        out[i_i] = out_stack[__idx_out_stack_0]
        outb[i_i - 1] = outb[i_i - 1] + y[i_i] * outb[i_i]
        yb[i_i] = yb[i_i] + out[i_i - 1] * outb[i_i]
        for i_j = i_m:-1:1
            __icse_11 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            __idx_out_stack_0 = __icse_11
            out[i_i] = out_stack[__idx_out_stack_0]
            __cse_12 = v * outb[i_i]
            vb = vb + __cse_12
            vb = vb + __cse_12
            __idx_v_stack_0 = __icse_11
            v = v_stack[__idx_v_stack_0]
            __oldb_2 = vb
            vb = 0.0
            xb[i_j] = xb[i_j] + y[i_i] * __oldb_2
            yb[i_i] = yb[i_i] + x[i_j] * __oldb_2
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
