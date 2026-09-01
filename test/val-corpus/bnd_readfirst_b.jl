function initstacks_bnd_readfirst_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return s_stack
end

function bnd_readfirst_b(x, xb, i_n, i_m, out, outb, s_stack)
    s = 0.0
    sb = 0.0
    s = 1.0
    for i_i = 1:i_n
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_1 = (i_i - 1) + 1
        s_stack[__idx_s_stack_1] = s
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_i] * x[i_j]
        end
        __idx_s_stack_5 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        s_stack[__idx_s_stack_5] = s
    end
    __idx_s_stack_2 = ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1
    s_stack[__idx_s_stack_2] = s
    __idx_s_stack_0 = ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1
    s = s_stack[__idx_s_stack_0]
    for i_i = i_n:-1:1
        __idx_s_stack_0 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        s = s_stack[__idx_s_stack_0]
        for i_j = 1:i_m
            s = s + x[i_i] * x[i_j]
            xb[i_i] = xb[i_i] + x[i_j] * sb
            xb[i_j] = xb[i_j] + x[i_i] * sb
        end
        __idx_s_stack_0 = (i_i - 1) + 1
        s = s_stack[__idx_s_stack_0]
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
