function initstacks_bnd_nested_only_b(i_m, i_n)
    v_stack = Vector{Float64}(undef, ((max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1) + 1)
    out_stack = Vector{Float64}(undef, max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1))
    return (v_stack, out_stack)
end

function bnd_nested_only_hv(x, xb, y, yb, i_n, i_m, out, outb, acc, accb, xd, xbd, yd, ybd, outd, outbd, accd, accbd, v_stack, out_stack)
    v_stack_d = Vector{Float64}(undef, length(v_stack))
    out_stack_d = Vector{Float64}(undef, length(out_stack))
    v = 0.0
    vb = 0.0
    vd = 0.0
    vbd = 0.0
    vd = 0.0
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            __icse_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            __idx_v_stack_0 = __icse_0
            v_stack_d[__idx_v_stack_0] = vd
            v_stack[__idx_v_stack_0] = v
            __hcse_0 = y[i_i]
            __hcse_1 = x[i_j]
            vd = __hcse_0 * xd[i_j] + __hcse_1 * yd[i_i]
            v = __hcse_1 * __hcse_0
            __idx_out_stack_3 = __icse_0
            __cse_1d = outd[i_i]
            __cse_1 = out[i_i]
            out_stack_d[__idx_out_stack_3] = __cse_1d
            out_stack[__idx_out_stack_3] = __cse_1
            __hcse_2 = v * vd
            outd[i_i] = __cse_1d + (__hcse_2 + __hcse_2)
            out[i_i] = __cse_1 + v * v
        end
        __icse_2 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        __idx_out_stack_1 = __icse_2
        __cse_3d = outd[i_i]
        __cse_3 = out[i_i]
        out_stack_d[__idx_out_stack_1] = __cse_3d
        out_stack[__idx_out_stack_1] = __cse_3
        __hcse_3 = y[i_i]
        __hcse_4 = out[i_i - 1]
        outd[i_i] = __cse_3d + (__hcse_3 * outd[i_i - 1] + __hcse_4 * yd[i_i])
        out[i_i] = __cse_3 + __hcse_4 * __hcse_3
        __idx_v_stack_4 = __icse_2
        v_stack_d[__idx_v_stack_4] = vd
        v_stack[__idx_v_stack_4] = v
    end
    __ihcse_5 = max(0, div(i_n - 2, 1) + 1)
    __icse_4 = __ihcse_5
    __ihcse_6 = max(0, div(i_m - 1, 1) + 1)
    __icse_5 = (__icse_4 * __ihcse_6 + __icse_4) + 1
    __idx_v_stack_2 = __icse_5
    v_stack_d[__idx_v_stack_2] = vd
    v_stack[__idx_v_stack_2] = v
    __hcse_7 = y[1]
    __hcse_8 = xd[1]
    __hcse_9 = x[1]
    __hcse_10 = yd[1]
    vd = __hcse_7 * __hcse_8 + __hcse_9 * __hcse_10
    v = __hcse_9 * __hcse_7
    __hcse_11 = v * vd
    accd[1] = accd[1] + (__hcse_11 + __hcse_11)
    acc[1] = acc[1] + v * v
    __idx_v_stack_6 = __icse_5 + 1
    v_stack_d[__idx_v_stack_6] = vd
    v_stack[__idx_v_stack_6] = v
    __icse_6 = __ihcse_5
    __icse_7 = (__icse_6 * __ihcse_6 + __icse_6) + 1
    __idx_v_stack_0 = __icse_7 + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    __hcse_12 = accb[1]
    __cse_8d = __hcse_12 * vd + v * accbd[1]
    __cse_8 = v * __hcse_12
    vbd = vbd + __cse_8d
    vb = vb + __cse_8
    vbd = vbd + __cse_8d
    vb = vb + __cse_8
    __idx_v_stack_0 = __icse_7
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    __oldb_2d = vbd
    __oldb_2 = vb
    vbd = 0.0
    vb = 0.0
    xbd[1] = xbd[1] + (__oldb_2 * __hcse_10 + __hcse_7 * __oldb_2d)
    xb[1] = xb[1] + __hcse_7 * __oldb_2
    ybd[1] = ybd[1] + (__oldb_2 * __hcse_8 + __hcse_9 * __oldb_2d)
    yb[1] = yb[1] + __hcse_9 * __oldb_2
    for i_i = i_n:-1:2
        __icse_9 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        __idx_v_stack_0 = __icse_9
        vd = v_stack_d[__idx_v_stack_0]
        v = v_stack[__idx_v_stack_0]
        __idx_out_stack_0 = __icse_9
        outd[i_i] = out_stack_d[__idx_out_stack_0]
        out[i_i] = out_stack[__idx_out_stack_0]
        __hcse_13 = outb[i_i]
        __hcse_14 = y[i_i]
        outbd[i_i - 1] = outbd[i_i - 1] + (__hcse_13 * yd[i_i] + __hcse_14 * outbd[i_i])
        outb[i_i - 1] = outb[i_i - 1] + __hcse_14 * __hcse_13
        __hcse_15 = outb[i_i]
        __hcse_16 = out[i_i - 1]
        ybd[i_i] = ybd[i_i] + (__hcse_15 * outd[i_i - 1] + __hcse_16 * outbd[i_i])
        yb[i_i] = yb[i_i] + __hcse_16 * __hcse_15
        for i_j = i_m:-1:1
            __icse_10 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            __idx_out_stack_0 = __icse_10
            outd[i_i] = out_stack_d[__idx_out_stack_0]
            out[i_i] = out_stack[__idx_out_stack_0]
            __hcse_17 = outb[i_i]
            __cse_11d = __hcse_17 * vd + v * outbd[i_i]
            __cse_11 = v * __hcse_17
            vbd = vbd + __cse_11d
            vb = vb + __cse_11
            vbd = vbd + __cse_11d
            vb = vb + __cse_11
            __idx_v_stack_0 = __icse_10
            vd = v_stack_d[__idx_v_stack_0]
            v = v_stack[__idx_v_stack_0]
            __oldb_2d = vbd
            __oldb_2 = vb
            vbd = 0.0
            vb = 0.0
            __hcse_18 = y[i_i]
            xbd[i_j] = xbd[i_j] + (__oldb_2 * yd[i_i] + __hcse_18 * __oldb_2d)
            xb[i_j] = xb[i_j] + __hcse_18 * __oldb_2
            __hcse_19 = x[i_j]
            ybd[i_i] = ybd[i_i] + (__oldb_2 * xd[i_j] + __hcse_19 * __oldb_2d)
            yb[i_i] = yb[i_i] + __hcse_19 * __oldb_2
        end
    end
    vbd = 0.0
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
