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
            __idx_s_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            s_stack_d[__idx_s_stack_0] = sd
            s_stack[__idx_s_stack_0] = s
            __cse_2 = y[i_i]
            __cse_3 = x[i_j]
            sd = sd + (__cse_2 * xd[i_j] + __cse_3 * yd[i_i])
            s = s + __cse_3 * __cse_2
        end
        __cse_4 = s * sd
        outd[i_i] = outd[i_i] + (__cse_4 + __cse_4)
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_2 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        s_stack_d[__idx_s_stack_2] = sd
        s_stack[__idx_s_stack_2] = s
    end
    __idx_s_stack_2 = (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1
    s_stack_d[__idx_s_stack_2] = sd
    s_stack[__idx_s_stack_2] = s
    __cse_5 = y[1]
    __cse_6 = xd[1]
    __cse_7 = x[1]
    __cse_8 = yd[1]
    sd = __cse_5 * __cse_6 + __cse_7 * __cse_8
    s = __cse_7 * __cse_5
    __cse_9 = s * sd
    accd[1] = accd[1] + (__cse_9 + __cse_9)
    acc[1] = acc[1] + s * s
    __idx_s_stack_6 = ((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1
    s_stack_d[__idx_s_stack_6] = sd
    s_stack[__idx_s_stack_6] = s
    __idx_s_stack_0 = ((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    __cse_10 = accb[1]
    __cse_0d = __cse_10 * sd + s * accbd[1]
    __cse_0 = s * __cse_10
    sbd = sbd + __cse_0d
    sb = sb + __cse_0
    sbd = sbd + __cse_0d
    sb = sb + __cse_0
    __idx_s_stack_0 = (max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    xbd[1] = xbd[1] + (sb * __cse_8 + __cse_5 * sbd)
    xb[1] = xb[1] + __cse_5 * sb
    ybd[1] = ybd[1] + (sb * __cse_6 + __cse_7 * sbd)
    yb[1] = yb[1] + __cse_7 * sb
    sbd = 0.0
    sb = 0.0
    for i_i = i_n:-1:1
        __idx_s_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __cse_11 = outb[i_i]
        __cse_1d = __cse_11 * sd + s * outbd[i_i]
        __cse_1 = s * __cse_11
        sbd = sbd + __cse_1d
        sb = sb + __cse_1
        sbd = sbd + __cse_1d
        sb = sb + __cse_1
        for i_j = i_m:-1:1
            __idx_s_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            __cse_12 = y[i_i]
            xbd[i_j] = xbd[i_j] + (sb * yd[i_i] + __cse_12 * sbd)
            xb[i_j] = xb[i_j] + __cse_12 * sb
            __cse_13 = x[i_j]
            ybd[i_i] = ybd[i_i] + (sb * xd[i_j] + __cse_13 * sbd)
            yb[i_i] = yb[i_i] + __cse_13 * sb
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
