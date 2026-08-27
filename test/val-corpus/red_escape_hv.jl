function initstacks_red_escape_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, ((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1)
    return s_stack
end

function red_escape_hv(x, xb, y, yb, i_n, i_m, out, outb, acc, accb, xd, xbd, yd, ybd, outd, outbd, accd, accbd, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    sd = 0.0
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            s_stack_d[((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1] = sd
            s_stack[((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1] = s
            sd = sd + (y[i_i] * xd[i_j] + x[i_j] * yd[i_i])
            s = s + x[i_j] * y[i_i]
        end
        outd[i_i] = outd[i_i] + (s * sd + s * sd)
        out[i_i] = out[i_i] + s * s
        s_stack_d[max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)] = sd
        s_stack[max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)] = s
    end
    s_stack_d[(max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1] = sd
    s_stack[(max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1] = s
    sd = y[1] * xd[1] + x[1] * yd[1]
    s = x[1] * y[1]
    accd[1] = accd[1] + (s * sd + s * sd)
    acc[1] = acc[1] + s * s
    s_stack_d[((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1] = sd
    s_stack[((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1] = s
    sd = s_stack_d[((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1]
    s = s_stack[((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1]
    sbd = sbd + (accb[1] * sd + s * accbd[1])
    sb = sb + s * accb[1]
    sbd = sbd + (accb[1] * sd + s * accbd[1])
    sb = sb + s * accb[1]
    sd = s_stack_d[(max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1]
    s = s_stack[(max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1]
    xbd[1] = xbd[1] + (sb * yd[1] + y[1] * sbd)
    xb[1] = xb[1] + y[1] * sb
    ybd[1] = ybd[1] + (sb * xd[1] + x[1] * sbd)
    yb[1] = yb[1] + x[1] * sb
    sbd = 0.0
    sb = 0.0
    for i_i = i_n:-1:1
        sd = s_stack_d[max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)]
        s = s_stack[max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)]
        sbd = sbd + (outb[i_i] * sd + s * outbd[i_i])
        sb = sb + s * outb[i_i]
        sbd = sbd + (outb[i_i] * sd + s * outbd[i_i])
        sb = sb + s * outb[i_i]
        for i_j = i_m:-1:1
            sd = s_stack_d[((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1]
            s = s_stack[((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1]
            xbd[i_j] = xbd[i_j] + (sb * yd[i_i] + y[i_i] * sbd)
            xb[i_j] = xb[i_j] + y[i_i] * sb
            ybd[i_i] = ybd[i_i] + (sb * xd[i_j] + x[i_j] * sbd)
            yb[i_i] = yb[i_i] + x[i_j] * sb
        end
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function red_escape(x, y, i_n, i_m, out, acc)
    s = 0.0
    for i_i = 1:i_n
        for i_j = 1:i_m
            s = s + x[i_j] * y[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
    s = x[1] * y[1]
    acc[1] = acc[1] + s * s
end
