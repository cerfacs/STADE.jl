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
        __cse_0 = x[i_i]
        __cse_4 = __cse_0 * xd[i_i]
        vd = __cse_4 + __cse_4
        v = __cse_0 * __cse_0
        __cse_5 = v * vd
        accd[i_i] = accd[i_i] + (__cse_5 + __cse_5)
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            __idx_v_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            v_stack_d[__idx_v_stack_0] = vd
            v_stack[__idx_v_stack_0] = v
            __cse_6 = y[i_j]
            __cse_7 = x[i_i]
            vd = __cse_6 * xd[i_i] + __cse_7 * yd[i_j]
            v = __cse_7 * __cse_6
            __cse_8 = v * vd
            outd[i_i] = outd[i_i] + (__cse_8 + __cse_8)
            out[i_i] = out[i_i] + v * v
        end
        __idx_v_stack_3 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        v_stack_d[__idx_v_stack_3] = vd
        v_stack[__idx_v_stack_3] = v
    end
    __idx_v_stack_2 = (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1
    v_stack_d[__idx_v_stack_2] = vd
    v_stack[__idx_v_stack_2] = v
    __idx_v_stack_0 = (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    for i_i = i_n:-1:1
        __idx_v_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        vd = v_stack_d[__idx_v_stack_0]
        v = v_stack[__idx_v_stack_0]
        for i_j = i_m:-1:1
            __cse_9 = outb[i_i]
            __cse_1d = __cse_9 * vd + v * outbd[i_i]
            __cse_1 = v * __cse_9
            vbd = vbd + __cse_1d
            vb = vb + __cse_1
            vbd = vbd + __cse_1d
            vb = vb + __cse_1
            __idx_v_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            vd = v_stack_d[__idx_v_stack_0]
            v = v_stack[__idx_v_stack_0]
            __cse_10 = y[i_j]
            xbd[i_i] = xbd[i_i] + (vb * yd[i_j] + __cse_10 * vbd)
            xb[i_i] = xb[i_i] + __cse_10 * vb
            __cse_11 = x[i_i]
            ybd[i_j] = ybd[i_j] + (vb * xd[i_i] + __cse_11 * vbd)
            yb[i_j] = yb[i_j] + __cse_11 * vb
            vbd = 0.0
            vb = 0.0
        end
        __cse_12 = accb[i_i]
        __cse_2d = __cse_12 * vd + v * accbd[i_i]
        __cse_2 = v * __cse_12
        vbd = vbd + __cse_2d
        vb = vb + __cse_2
        vbd = vbd + __cse_2d
        vb = vb + __cse_2
        __cse_13 = x[i_i]
        __cse_3d = vb * xd[i_i] + __cse_13 * vbd
        __cse_3 = __cse_13 * vb
        xbd[i_i] = xbd[i_i] + __cse_3d
        xb[i_i] = xb[i_i] + __cse_3
        xbd[i_i] = xbd[i_i] + __cse_3d
        xb[i_i] = xb[i_i] + __cse_3
        vbd = 0.0
        vb = 0.0
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
