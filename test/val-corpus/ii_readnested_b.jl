function initstacks_ii_readnested_b(i_m, i_n)
    v_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return v_stack
end

function ii_readnested_b(x, xb, y, yb, i_n, i_m, out, outb, acc, accb, v_stack)
    v = 0.0
    vb = 0.0
    v = 1.0
    for i_i = 1:i_n
        __cse_0 = x[i_i]
        v = __cse_0 * __cse_0
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            __idx_v_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            v_stack[__idx_v_stack_0] = v
            v = x[i_i] * y[i_j]
            out[i_i] = out[i_i] + v * v
        end
        __idx_v_stack_3 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        v_stack[__idx_v_stack_3] = v
    end
    __icse_1 = max(0, div(i_n - 1, 1) + 1)
    __icse_2 = (__icse_1 * max(0, div(i_m - 1, 1) + 1) + __icse_1) + 1
    __idx_v_stack_2 = __icse_2
    v_stack[__idx_v_stack_2] = v
    __idx_v_stack_0 = __icse_2
    v = v_stack[__idx_v_stack_0]
    for i_i = i_n:-1:1
        __idx_v_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        v = v_stack[__idx_v_stack_0]
        for i_j = i_m:-1:1
            __cse_3 = v * outb[i_i]
            vb = vb + __cse_3
            vb = vb + __cse_3
            __idx_v_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            v = v_stack[__idx_v_stack_0]
            __oldb_2 = vb
            vb = 0.0
            xb[i_i] = xb[i_i] + y[i_j] * __oldb_2
            yb[i_j] = yb[i_j] + x[i_i] * __oldb_2
        end
        __cse_4 = v * accb[i_i]
        vb = vb + __cse_4
        vb = vb + __cse_4
        __oldb_0 = vb
        vb = 0.0
        __cse_5 = x[i_i] * __oldb_0
        xb[i_i] = xb[i_i] + __cse_5
        xb[i_i] = xb[i_i] + __cse_5
    end
    vb = 0.0
    return nothing
end

function ii_readnested(x, y, i_n, i_m, out, acc)
    v = 1.0
    for i_i = 1:i_n
        v = x[i_i] * x[i_i]
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            v = x[i_i] * y[i_j]
            out[i_i] = out[i_i] + v * v
        end
    end
end
