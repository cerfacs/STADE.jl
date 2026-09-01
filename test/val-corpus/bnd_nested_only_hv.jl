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
            __idx_v_stack_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            v_stack_d[__idx_v_stack_0] = vd
            v_stack[__idx_v_stack_0] = v
            __cse_4 = y[i_i]
            __cse_5 = x[i_j]
            vd = __cse_4 * xd[i_j] + __cse_5 * yd[i_i]
            v = __cse_5 * __cse_4
            __idx_out_stack_3 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            __cse_0d = outd[i_i]
            __cse_0 = out[i_i]
            out_stack_d[__idx_out_stack_3] = __cse_0d
            out_stack[__idx_out_stack_3] = __cse_0
            __cse_6 = v * vd
            outd[i_i] = __cse_0d + (__cse_6 + __cse_6)
            out[i_i] = __cse_0 + v * v
        end
        __idx_out_stack_1 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        __cse_1d = outd[i_i]
        __cse_1 = out[i_i]
        out_stack_d[__idx_out_stack_1] = __cse_1d
        out_stack[__idx_out_stack_1] = __cse_1
        __cse_7 = y[i_i]
        __cse_8 = out[i_i - 1]
        outd[i_i] = __cse_1d + (__cse_7 * outd[i_i - 1] + __cse_8 * yd[i_i])
        out[i_i] = __cse_1 + __cse_8 * __cse_7
        __idx_v_stack_4 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        v_stack_d[__idx_v_stack_4] = vd
        v_stack[__idx_v_stack_4] = v
    end
    __idx_v_stack_2 = (max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1
    v_stack_d[__idx_v_stack_2] = vd
    v_stack[__idx_v_stack_2] = v
    __cse_9 = y[1]
    __cse_10 = xd[1]
    __cse_11 = x[1]
    __cse_12 = yd[1]
    vd = __cse_9 * __cse_10 + __cse_11 * __cse_12
    v = __cse_11 * __cse_9
    __cse_13 = v * vd
    accd[1] = accd[1] + (__cse_13 + __cse_13)
    acc[1] = acc[1] + v * v
    __idx_v_stack_6 = ((max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1) + 1
    v_stack_d[__idx_v_stack_6] = vd
    v_stack[__idx_v_stack_6] = v
    __idx_v_stack_0 = ((max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1) + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    __cse_14 = accb[1]
    __cse_2d = __cse_14 * vd + v * accbd[1]
    __cse_2 = v * __cse_14
    vbd = vbd + __cse_2d
    vb = vb + __cse_2
    vbd = vbd + __cse_2d
    vb = vb + __cse_2
    __idx_v_stack_0 = (max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    xbd[1] = xbd[1] + (vb * __cse_12 + __cse_9 * vbd)
    xb[1] = xb[1] + __cse_9 * vb
    ybd[1] = ybd[1] + (vb * __cse_10 + __cse_11 * vbd)
    yb[1] = yb[1] + __cse_11 * vb
    vbd = 0.0
    vb = 0.0
    for i_i = i_n:-1:2
        __idx_v_stack_0 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        vd = v_stack_d[__idx_v_stack_0]
        v = v_stack[__idx_v_stack_0]
        __idx_out_stack_0 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        outd[i_i] = out_stack_d[__idx_out_stack_0]
        out[i_i] = out_stack[__idx_out_stack_0]
        __cse_15 = outb[i_i]
        __cse_16 = y[i_i]
        outbd[i_i - 1] = outbd[i_i - 1] + (__cse_15 * yd[i_i] + __cse_16 * outbd[i_i])
        outb[i_i - 1] = outb[i_i - 1] + __cse_16 * __cse_15
        __cse_17 = outb[i_i]
        __cse_18 = out[i_i - 1]
        ybd[i_i] = ybd[i_i] + (__cse_17 * outd[i_i - 1] + __cse_18 * outbd[i_i])
        yb[i_i] = yb[i_i] + __cse_18 * __cse_17
        for i_j = i_m:-1:1
            __idx_out_stack_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            outd[i_i] = out_stack_d[__idx_out_stack_0]
            out[i_i] = out_stack[__idx_out_stack_0]
            __cse_19 = outb[i_i]
            __cse_3d = __cse_19 * vd + v * outbd[i_i]
            __cse_3 = v * __cse_19
            vbd = vbd + __cse_3d
            vb = vb + __cse_3
            vbd = vbd + __cse_3d
            vb = vb + __cse_3
            __idx_v_stack_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            vd = v_stack_d[__idx_v_stack_0]
            v = v_stack[__idx_v_stack_0]
            __cse_20 = y[i_i]
            xbd[i_j] = xbd[i_j] + (vb * yd[i_i] + __cse_20 * vbd)
            xb[i_j] = xb[i_j] + __cse_20 * vb
            __cse_21 = x[i_j]
            ybd[i_i] = ybd[i_i] + (vb * xd[i_j] + __cse_21 * vbd)
            yb[i_i] = yb[i_i] + __cse_21 * vb
            vbd = 0.0
            vb = 0.0
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
