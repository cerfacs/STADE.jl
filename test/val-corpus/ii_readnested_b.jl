function initstacks_ii_readnested_b(i_m, i_n)
    v_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return v_stack
end

function ii_readnested_b(x, xb, y, yb, i_n, i_m, out, outb, acc, accb, v_stack)
    v = 0.0
    vb = 0.0
    v = 1.0
    for i_i = 1:i_n
        v = x[i_i] * x[i_i]
        acc[i_i] = acc[i_i] + v * v
        for i_j = 1:i_m
            v_stack[((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1] = v
            v = x[i_i] * y[i_j]
            out[i_i] = out[i_i] + v * v
        end
        v_stack[max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)] = v
    end
    v_stack[(max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1] = v
    v = v_stack[(max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1]
    for i_i = i_n:-1:1
        v = v_stack[max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)]
        for i_j = i_m:-1:1
            vb = vb + v * outb[i_i]
            vb = vb + v * outb[i_i]
            v = v_stack[((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1]
            xb[i_i] = xb[i_i] + y[i_j] * vb
            yb[i_j] = yb[i_j] + x[i_i] * vb
            vb = 0.0
        end
        vb = vb + v * accb[i_i]
        vb = vb + v * accb[i_i]
        xb[i_i] = xb[i_i] + x[i_i] * vb
        xb[i_i] = xb[i_i] + x[i_i] * vb
        vb = 0.0
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
