function initstacks_ii_readbefore_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return s_stack
end

function ii_readbefore_b(x, xb, i_n, i_m, out, outb, s_stack)
    s = 0.0
    sb = 0.0
    __cse_0 = x[1]
    s = __cse_0 * __cse_0
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        __idx_s_stack_0 = (i_i - 1) + 1
        s_stack[__idx_s_stack_0] = s
        s = 0.0
        for i_j = 1:i_m
            s = s + x[i_j] * x[i_i]
        end
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_5 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        s_stack[__idx_s_stack_5] = s
    end
    __idx_s_stack_3 = ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1
    s_stack[__idx_s_stack_3] = s
    __idx_s_stack_0 = ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1
    s = s_stack[__idx_s_stack_0]
    for i_i = i_n:-1:1
        __idx_s_stack_0 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        s = s_stack[__idx_s_stack_0]
        __cse_1 = s * outb[i_i]
        sb = sb + __cse_1
        sb = sb + __cse_1
        for i_j = 1:i_m
            __cse_2 = x[i_j]
            __cse_3 = x[i_i]
            s = s + __cse_2 * __cse_3
            xb[i_j] = xb[i_j] + __cse_3 * sb
            xb[i_i] = xb[i_i] + __cse_2 * sb
        end
        __idx_s_stack_0 = (i_i - 1) + 1
        s = s_stack[__idx_s_stack_0]
        sb = 0.0
    end
    __cse_4 = s * outb[1]
    sb = sb + __cse_4
    sb = sb + __cse_4
    __cse_5 = x[1] * sb
    xb[1] = xb[1] + __cse_5
    xb[1] = xb[1] + __cse_5
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
