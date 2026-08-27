function initstacks_ii_readbefore_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return s_stack
end

function ii_readbefore_b(x, xb, i_n, i_m, out, outb, s_stack)
    s = 0.0
    sb = 0.0
    s = x[1] * x[1]
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        s_stack[(i_i - 1) + 1] = s
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_j] * x[i_i]
        end
        out[i_i] = out[i_i] + s * s
        s_stack[(max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)] = s
    end
    s_stack[((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1] = s
    s = s_stack[((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1]
    for i_i = i_n:-1:1
        s = s_stack[(max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)]
        sb = sb + s * outb[i_i]
        sb = sb + s * outb[i_i]
        for i_j = 1:i_m
            s = s + x[i_j] * x[i_i]
            xb[i_j] = xb[i_j] + x[i_i] * sb
            xb[i_i] = xb[i_i] + x[i_j] * sb
        end
        s = s_stack[(i_i - 1) + 1]
        sb = 0.0
    end
    sb = sb + s * outb[1]
    sb = sb + s * outb[1]
    xb[1] = xb[1] + x[1] * sb
    xb[1] = xb[1] + x[1] * sb
    sb = 0.0
    return nothing
end

function ii_readbefore(x, i_n, i_m, out)
    s = x[1] * x[1]
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_j] * x[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
end
