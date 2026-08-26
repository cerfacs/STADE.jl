function initstacks_bnd_readfirst_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, (((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + (div(i_n - 1, 1) + 1)) + 1)
    return s_stack
end

function bnd_readfirst_hv(x, xb, i_n, i_m, out, outb, xd, xbd, outd, outbd, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    sd = 0.0
    s = 1.0
    for i_i = 1:i_n
        outd[i_i] = outd[i_i] + (s * sd + s * sd)
        out[i_i] = out[i_i] + s * s
        s_stack_d[(i_i - 1) + 1] = sd
        s_stack[(i_i - 1) + 1] = s
        sd = 0.0
        s = 0.0
        for i_j = 1:i_m
            sd = sd + (x[i_j] * xd[i_i] + x[i_i] * xd[i_j])
            s = s + x[i_i] * x[i_j]
        end
        s_stack_d[((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)] = sd
        s_stack[((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)] = s
    end
    s_stack_d[(((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + (div(i_n - 1, 1) + 1)) + 1] = sd
    s_stack[(((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + (div(i_n - 1, 1) + 1)) + 1] = s
    sd = s_stack_d[(((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + (div(i_n - 1, 1) + 1)) + 1]
    s = s_stack[(((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + (div(i_n - 1, 1) + 1)) + 1]
    for i_i = i_n:-1:1
        sd = s_stack_d[((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)]
        s = s_stack[((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)]
        for i_j = 1:i_m
            sd = sd + (x[i_j] * xd[i_i] + x[i_i] * xd[i_j])
            s = s + x[i_i] * x[i_j]
            xbd[i_i] = xbd[i_i] + (sb * xd[i_j] + x[i_j] * sbd)
            xb[i_i] = xb[i_i] + x[i_j] * sb
            xbd[i_j] = xbd[i_j] + (sb * xd[i_i] + x[i_i] * sbd)
            xb[i_j] = xb[i_j] + x[i_i] * sb
        end
        sd = s_stack_d[(i_i - 1) + 1]
        s = s_stack[(i_i - 1) + 1]
        sbd = 0.0
        sb = 0.0
        sbd = sbd + (outb[i_i] * sd + s * outbd[i_i])
        sb = sb + s * outb[i_i]
        sbd = sbd + (outb[i_i] * sd + s * outbd[i_i])
        sb = sb + s * outb[i_i]
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function bnd_readfirst(x, i_n, i_m, out)
    s = 1.0
    for i_i = 1:i_n
        out[i_i] = out[i_i] + s * s
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_i] * x[i_j]
        end
    end
end
