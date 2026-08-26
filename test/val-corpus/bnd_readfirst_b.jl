function initstacks_bnd_readfirst_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, (((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + (div(i_n - 1, 1) + 1)) + 1)
    return s_stack
end

function bnd_readfirst_b(x, xb, i_n, i_m, out, outb, s_stack)
    s = 0.0
    sb = 0.0
    s = 1.0
    for i_i = 1:i_n
        out[i_i] = out[i_i] + s * s
        s_stack[(i_i - 1) + 1] = s
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_i] * x[i_j]
        end
        s_stack[((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)] = s
    end
    s_stack[(((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + (div(i_n - 1, 1) + 1)) + 1] = s
    s = s_stack[(((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + (div(i_n - 1, 1) + 1)) + 1]
    for i_i = i_n:-1:1
        s = s_stack[((div(i_n - 1, 1) + 1) + (div(i_n - 1, 1) + 1) * (div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)]
        for i_j = 1:i_m
            s = s + x[i_i] * x[i_j]
            xb[i_i] = xb[i_i] + x[i_j] * sb
            xb[i_j] = xb[i_j] + x[i_i] * sb
        end
        s = s_stack[(i_i - 1) + 1]
        sb = 0.0
        sb = sb + s * outb[i_i]
        sb = sb + s * outb[i_i]
    end
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
