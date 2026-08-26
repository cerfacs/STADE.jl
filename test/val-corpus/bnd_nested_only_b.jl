function initstacks_bnd_nested_only_b(i_m, i_n)
    v_stack = Vector{Float64}(undef, (((div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + (div(i_n - 2, 1) + 1)) + 1) + 1)
    out_stack = Vector{Float64}(undef, (div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + (div(i_n - 2, 1) + 1))
    return (v_stack, out_stack)
end

function bnd_nested_only_b(x, xb, y, yb, i_n, i_m, out, outb, acc, accb, v_stack, out_stack)
    v = 0.0
    vb = 0.0
    v = 0.0
    for i_i = 2:i_n
        for i_j = 1:i_m
            v_stack[((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1] = v
            v = x[i_j] * y[i_i]
            out_stack[((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1] = out[i_i]
            out[i_i] = out[i_i] + v * v
        end
        out_stack[(div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)] = out[i_i]
        out[i_i] = out[i_i] + out[i_i - 1] * y[i_i]
        v_stack[(div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)] = v
    end
    v_stack[((div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + (div(i_n - 2, 1) + 1)) + 1] = v
    v = x[1] * y[1]
    acc[1] = acc[1] + v * v
    v_stack[(((div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + (div(i_n - 2, 1) + 1)) + 1) + 1] = v
    v = v_stack[(((div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + (div(i_n - 2, 1) + 1)) + 1) + 1]
    vb = vb + v * accb[1]
    vb = vb + v * accb[1]
    v = v_stack[((div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + (div(i_n - 2, 1) + 1)) + 1]
    xb[1] = xb[1] + y[1] * vb
    yb[1] = yb[1] + x[1] * vb
    vb = 0.0
    for i_i = i_n:-1:2
        v = v_stack[(div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)]
        out[i_i] = out_stack[(div(i_n - 2, 1) + 1) * (div(i_m - 1, 1) + 1) + ((i_i - 2) + 1)]
        outb[i_i - 1] = outb[i_i - 1] + y[i_i] * outb[i_i]
        yb[i_i] = yb[i_i] + out[i_i - 1] * outb[i_i]
        for i_j = i_m:-1:1
            out[i_i] = out_stack[((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1]
            vb = vb + v * outb[i_i]
            vb = vb + v * outb[i_i]
            v = v_stack[((i_i - 2) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1]
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
