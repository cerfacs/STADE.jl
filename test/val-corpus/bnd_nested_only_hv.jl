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
            vd = y[i_i] * xd[i_j] + x[i_j] * yd[i_i]
            v = x[i_j] * y[i_i]
            __idx_out_stack_3 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            out_stack_d[__idx_out_stack_3] = outd[i_i]
            out_stack[__idx_out_stack_3] = out[i_i]
            outd[i_i] = outd[i_i] + (v * vd + v * vd)
            out[i_i] = out[i_i] + v * v
        end
        __idx_out_stack_1 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        out_stack_d[__idx_out_stack_1] = outd[i_i]
        out_stack[__idx_out_stack_1] = out[i_i]
        outd[i_i] = outd[i_i] + (y[i_i] * outd[i_i - 1] + out[i_i - 1] * yd[i_i])
        out[i_i] = out[i_i] + out[i_i - 1] * y[i_i]
        __idx_v_stack_4 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        v_stack_d[__idx_v_stack_4] = vd
        v_stack[__idx_v_stack_4] = v
    end
    __idx_v_stack_2 = (max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1
    v_stack_d[__idx_v_stack_2] = vd
    v_stack[__idx_v_stack_2] = v
    vd = y[1] * xd[1] + x[1] * yd[1]
    v = x[1] * y[1]
    accd[1] = accd[1] + (v * vd + v * vd)
    acc[1] = acc[1] + v * v
    __idx_v_stack_6 = ((max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1) + 1
    v_stack_d[__idx_v_stack_6] = vd
    v_stack[__idx_v_stack_6] = v
    __idx_v_stack_0 = ((max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1) + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    vbd = vbd + (accb[1] * vd + v * accbd[1])
    vb = vb + v * accb[1]
    vbd = vbd + (accb[1] * vd + v * accbd[1])
    vb = vb + v * accb[1]
    __idx_v_stack_0 = (max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 2, 1) + 1)) + 1
    vd = v_stack_d[__idx_v_stack_0]
    v = v_stack[__idx_v_stack_0]
    xbd[1] = xbd[1] + (vb * yd[1] + y[1] * vbd)
    xb[1] = xb[1] + y[1] * vb
    ybd[1] = ybd[1] + (vb * xd[1] + x[1] * vbd)
    yb[1] = yb[1] + x[1] * vb
    vbd = 0.0
    vb = 0.0
    for i_i = i_n:-1:2
        __idx_v_stack_0 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        vd = v_stack_d[__idx_v_stack_0]
        v = v_stack[__idx_v_stack_0]
        __idx_out_stack_0 = max(0, div(i_n - 2, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)
        outd[i_i] = out_stack_d[__idx_out_stack_0]
        out[i_i] = out_stack[__idx_out_stack_0]
        outbd[i_i - 1] = outbd[i_i - 1] + (outb[i_i] * yd[i_i] + y[i_i] * outbd[i_i])
        outb[i_i - 1] = outb[i_i - 1] + y[i_i] * outb[i_i]
        ybd[i_i] = ybd[i_i] + (outb[i_i] * outd[i_i - 1] + out[i_i - 1] * outbd[i_i])
        yb[i_i] = yb[i_i] + out[i_i - 1] * outb[i_i]
        for i_j = i_m:-1:1
            __idx_out_stack_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            outd[i_i] = out_stack_d[__idx_out_stack_0]
            out[i_i] = out_stack[__idx_out_stack_0]
            vbd = vbd + (outb[i_i] * vd + v * outbd[i_i])
            vb = vb + v * outb[i_i]
            vbd = vbd + (outb[i_i] * vd + v * outbd[i_i])
            vb = vb + v * outb[i_i]
            __idx_v_stack_0 = ((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            vd = v_stack_d[__idx_v_stack_0]
            v = v_stack[__idx_v_stack_0]
            xbd[i_j] = xbd[i_j] + (vb * yd[i_i] + y[i_i] * vbd)
            xb[i_j] = xb[i_j] + y[i_i] * vb
            ybd[i_i] = ybd[i_i] + (vb * xd[i_j] + x[i_j] * vbd)
            yb[i_i] = yb[i_i] + x[i_j] * vb
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
