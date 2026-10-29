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
            __hcse_0 = y[i_i]
            __hcse_1 = x[i_j]
            sd = sd + (__hcse_0 * xd[i_j] + __hcse_1 * yd[i_i])
            s = s + __hcse_1 * __hcse_0
        end
        __hcse_2 = s * sd
        outd[i_i] = outd[i_i] + (__hcse_2 + __hcse_2)
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_2 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        s_stack_d[__idx_s_stack_2] = sd
        s_stack[__idx_s_stack_2] = s
    end
    __ihcse_3 = max(0, div(i_n - 1, 1) + 1)
    __icse_0 = __ihcse_3
    __ihcse_4 = max(0, div(i_m - 1, 1) + 1)
    __icse_1 = (__icse_0 * __ihcse_4 + __icse_0) + 1
    __idx_s_stack_2 = __icse_1
    s_stack_d[__idx_s_stack_2] = sd
    s_stack[__idx_s_stack_2] = s
    __hcse_5 = y[1]
    __hcse_6 = xd[1]
    __hcse_7 = x[1]
    __hcse_8 = yd[1]
    sd = __hcse_5 * __hcse_6 + __hcse_7 * __hcse_8
    s = __hcse_7 * __hcse_5
    __hcse_9 = s * sd
    accd[1] = accd[1] + (__hcse_9 + __hcse_9)
    acc[1] = acc[1] + s * s
    __idx_s_stack_6 = __icse_1 + 1
    s_stack_d[__idx_s_stack_6] = sd
    s_stack[__idx_s_stack_6] = s
    __icse_2 = __ihcse_3
    __icse_3 = (__icse_2 * __ihcse_4 + __icse_2) + 1
    __idx_s_stack_0 = __icse_3 + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    __hcse_10 = accb[1]
    __cse_4d = __hcse_10 * sd + s * accbd[1]
    __cse_4 = s * __hcse_10
    sbd = sbd + __cse_4d
    sb = sb + __cse_4
    sbd = sbd + __cse_4d
    sb = sb + __cse_4
    __idx_s_stack_0 = __icse_3
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    __oldb_2d = sbd
    __oldb_2 = sb
    sbd = 0.0
    sb = 0.0
    xbd[1] = xbd[1] + (__oldb_2 * __hcse_8 + __hcse_5 * __oldb_2d)
    xb[1] = xb[1] + __hcse_5 * __oldb_2
    ybd[1] = ybd[1] + (__oldb_2 * __hcse_6 + __hcse_7 * __oldb_2d)
    yb[1] = yb[1] + __hcse_7 * __oldb_2
    for i_i = i_n:-1:1
        __idx_s_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1) + ((i_i - 1) + 1)
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __hcse_11 = outb[i_i]
        __cse_5d = __hcse_11 * sd + s * outbd[i_i]
        __cse_5 = s * __hcse_11
        sbd = sbd + __cse_5d
        sb = sb + __cse_5
        sbd = sbd + __cse_5d
        sb = sb + __cse_5
        for i_j = i_m:-1:1
            __idx_s_stack_0 = ((i_i - 1) * (div(i_m - 1, 1) + 1) + (i_j - 1)) + 1
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            __hcse_12 = y[i_i]
            xbd[i_j] = xbd[i_j] + (sb * yd[i_i] + __hcse_12 * sbd)
            xb[i_j] = xb[i_j] + __hcse_12 * sb
            __hcse_13 = x[i_j]
            ybd[i_i] = ybd[i_i] + (sb * xd[i_j] + __hcse_13 * sbd)
            yb[i_i] = yb[i_i] + __hcse_13 * sb
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
