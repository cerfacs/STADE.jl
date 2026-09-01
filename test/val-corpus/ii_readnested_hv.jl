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
        vd = x[i_i] * xd[i_i] + x[i_i] * xd[i_i]
        v = x[i_i] * x[i_i]
        accd[i_i] = accd[i_i] + (v * vd + v * vd)
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            __idx_v_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            v_stack_d[__idx_v_stack_0] = vd
            v_stack[__idx_v_stack_0] = v
            vd = y[i_j] * xd[i_i] + x[i_i] * yd[i_j]
            v = x[i_i] * y[i_j]
            outd[i_i] = outd[i_i] + (v * vd + v * vd)
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
            vbd = vbd + (outb[i_i] * vd + v * outbd[i_i])
            vb = vb + v * outb[i_i]
            vbd = vbd + (outb[i_i] * vd + v * outbd[i_i])
            vb = vb + v * outb[i_i]
            __idx_v_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            vd = v_stack_d[__idx_v_stack_0]
            v = v_stack[__idx_v_stack_0]
            xbd[i_i] = xbd[i_i] + (vb * yd[i_j] + y[i_j] * vbd)
            xb[i_i] = xb[i_i] + y[i_j] * vb
            ybd[i_j] = ybd[i_j] + (vb * xd[i_i] + x[i_i] * vbd)
            yb[i_j] = yb[i_j] + x[i_i] * vb
            vbd = 0.0
            vb = 0.0
        end
        vbd = vbd + (accb[i_i] * vd + v * accbd[i_i])
        vb = vb + v * accb[i_i]
        vbd = vbd + (accb[i_i] * vd + v * accbd[i_i])
        vb = vb + v * accb[i_i]
        xbd[i_i] = xbd[i_i] + (vb * xd[i_i] + x[i_i] * vbd)
        xb[i_i] = xb[i_i] + x[i_i] * vb
        xbd[i_i] = xbd[i_i] + (vb * xd[i_i] + x[i_i] * vbd)
        xb[i_i] = xb[i_i] + x[i_i] * vb
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
