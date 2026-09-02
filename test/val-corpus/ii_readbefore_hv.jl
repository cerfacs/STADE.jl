function initstacks_ii_readbefore_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return s_stack
end

function ii_readbefore_hv(x, xb, i_n, i_m, out, outb, xd, xbd, outd, outbd, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    __cse_0d = xd[1]
    __cse_0 = x[1]
    __hcse_0 = __cse_0 * __cse_0d
    sd = __hcse_0 + __hcse_0
    s = __cse_0 * __cse_0
    __hcse_1 = s * sd
    outd[1] = outd[1] + (__hcse_1 + __hcse_1)
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        __idx_s_stack_0 = (i_i - 1) + 1
        s_stack_d[__idx_s_stack_0] = sd
        s_stack[__idx_s_stack_0] = s
        sd = 0.0
        s = 0.0
        for i_j = 1:i_m
            __hcse_2 = x[i_i]
            __hcse_3 = x[i_j]
            sd = sd + (__hcse_2 * xd[i_j] + __hcse_3 * xd[i_i])
            s = s + __hcse_3 * __hcse_2
        end
        __hcse_4 = s * sd
        outd[i_i] = outd[i_i] + (__hcse_4 + __hcse_4)
        out[i_i] = out[i_i] + s * s
        __icse_1 = max(0, div(i_n - 1, 1) + 1)
        __idx_s_stack_5 = (__icse_1 + __icse_1 * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        s_stack_d[__idx_s_stack_5] = sd
        s_stack[__idx_s_stack_5] = s
    end
    __ihcse_5 = max(0, div(i_n - 1, 1) + 1)
    __icse_2 = __ihcse_5
    __ihcse_6 = max(0, div(i_m - 1, 1) + 1)
    __idx_s_stack_3 = ((__icse_2 + __icse_2 * __ihcse_6) + __icse_2) + 1
    s_stack_d[__idx_s_stack_3] = sd
    s_stack[__idx_s_stack_3] = s
    __icse_3 = __ihcse_5
    __idx_s_stack_0 = ((__icse_3 + __icse_3 * __ihcse_6) + __icse_3) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_i = i_n:-1:1
        __icse_4 = max(0, div(i_n - 1, 1) + 1)
        __idx_s_stack_0 = (__icse_4 + __icse_4 * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __hcse_7 = outb[i_i]
        __cse_5d = __hcse_7 * sd + s * outbd[i_i]
        __cse_5 = s * __hcse_7
        sbd = sbd + __cse_5d
        sb = sb + __cse_5
        sbd = sbd + __cse_5d
        sb = sb + __cse_5
        for i_j = 1:i_m
            __cse_6d = xd[i_j]
            __cse_6 = x[i_j]
            __cse_7d = xd[i_i]
            __cse_7 = x[i_i]
            sd = sd + (__cse_7 * __cse_6d + __cse_6 * __cse_7d)
            s = s + __cse_6 * __cse_7
            xbd[i_j] = xbd[i_j] + (sb * __cse_7d + __cse_7 * sbd)
            xb[i_j] = xb[i_j] + __cse_7 * sb
            xbd[i_i] = xbd[i_i] + (sb * __cse_6d + __cse_6 * sbd)
            xb[i_i] = xb[i_i] + __cse_6 * sb
        end
        __idx_s_stack_0 = (i_i - 1) + 1
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        sbd = 0.0
        sb = 0.0
    end
    __hcse_8 = outb[1]
    __cse_8d = __hcse_8 * sd + s * outbd[1]
    __cse_8 = s * __hcse_8
    sbd = sbd + __cse_8d
    sb = sb + __cse_8
    sbd = sbd + __cse_8d
    sb = sb + __cse_8
    __oldb_0d = sbd
    __oldb_0 = sb
    sbd = 0.0
    sb = 0.0
    __hcse_9 = x[1]
    __cse_9d = __oldb_0 * xd[1] + __hcse_9 * __oldb_0d
    __cse_9 = __hcse_9 * __oldb_0
    xbd[1] = xbd[1] + __cse_9d
    xb[1] = xb[1] + __cse_9
    xbd[1] = xbd[1] + __cse_9d
    xb[1] = xb[1] + __cse_9
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
