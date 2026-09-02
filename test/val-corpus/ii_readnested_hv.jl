function initstacks_ii_readnested_b(i_m, i_n)
    v_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return v_stack
end

function ii_readnested_hv(x, xb, y, yb, i_n, i_m, out, outb, acc, accb, xd, xbd, yd, ybd, outd, outbd, accd, accbd, v_stack)
    v_stack_d = Vector{Float64}(undef, length(v_stack))
    v = 0.0
    vb = 0.0
    vd = 0.0
    vbd = 0.0
    vd = 0.0
    v = 1.0
    for i_i = 1:i_n
        __cse_0d = xd[i_i]
        __cse_0 = x[i_i]
        __hcse_0 = __cse_0 * __cse_0d
        vd = __hcse_0 + __hcse_0
        v = __cse_0 * __cse_0
        __hcse_1 = v * vd
        accd[i_i] = accd[i_i] + (__hcse_1 + __hcse_1)
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            __idx_v_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            v_stack_d[__idx_v_stack_0] = vd
            v_stack[__idx_v_stack_0] = v
            __hcse_2 = y[i_j]
            __hcse_3 = x[i_i]
            vd = __hcse_2 * xd[i_i] + __hcse_3 * yd[i_j]
            v = __hcse_3 * __hcse_2
            __hcse_4 = v * vd
            outd[i_i] = outd[i_i] + (__hcse_4 + __hcse_4)
            out[i_i] = out[i_i] + v * v
        end
        __idx_v_stack_3 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        v_stack_d[__idx_v_stack_3] = vd
        v_stack[__idx_v_stack_3] = v
    end
    __ihcse_5 = max(0, div(i_n - 1, 1) + 1)
    __icse_1 = __ihcse_5
    __ihcse_6 = max(0, div(i_m - 1, 1) + 1)
    __idx_v_stack_2 = (__icse_1 * __ihcse_6 + __icse_1) + 1
    v_stack_d[__idx_v_stack_2] = vd
    v_stack[__idx_v_stack_2] = v
    __icse_2 = __ihcse_5
    __idx_v_stack_0 = (__icse_2 * __ihcse_6 + __icse_2) + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    for i_i = i_n:-1:1
        __idx_v_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        vd = v_stack_d[__idx_v_stack_0]
        v = v_stack[__idx_v_stack_0]
        for i_j = i_m:-1:1
            __hcse_7 = outb[i_i]
            __cse_3d = __hcse_7 * vd + v * outbd[i_i]
            __cse_3 = v * __hcse_7
            vbd = vbd + __cse_3d
            vb = vb + __cse_3
            vbd = vbd + __cse_3d
            vb = vb + __cse_3
            __idx_v_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            vd = v_stack_d[__idx_v_stack_0]
            v = v_stack[__idx_v_stack_0]
            __oldb_2d = vbd
            __oldb_2 = vb
            vbd = 0.0
            vb = 0.0
            __hcse_8 = y[i_j]
            xbd[i_i] = xbd[i_i] + (__oldb_2 * yd[i_j] + __hcse_8 * __oldb_2d)
            xb[i_i] = xb[i_i] + __hcse_8 * __oldb_2
            __hcse_9 = x[i_i]
            ybd[i_j] = ybd[i_j] + (__oldb_2 * xd[i_i] + __hcse_9 * __oldb_2d)
            yb[i_j] = yb[i_j] + __hcse_9 * __oldb_2
        end
        __hcse_10 = accb[i_i]
        __cse_4d = __hcse_10 * vd + v * accbd[i_i]
        __cse_4 = v * __hcse_10
        vbd = vbd + __cse_4d
        vb = vb + __cse_4
        vbd = vbd + __cse_4d
        vb = vb + __cse_4
        __oldb_0d = vbd
        __oldb_0 = vb
        vbd = 0.0
        vb = 0.0
        __hcse_11 = x[i_i]
        __cse_5d = __oldb_0 * xd[i_i] + __hcse_11 * __oldb_0d
        __cse_5 = __hcse_11 * __oldb_0
        xbd[i_i] = xbd[i_i] + __cse_5d
        xb[i_i] = xb[i_i] + __cse_5
        xbd[i_i] = xbd[i_i] + __cse_5d
        xb[i_i] = xb[i_i] + __cse_5
    end
    vbd = 0.0
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
