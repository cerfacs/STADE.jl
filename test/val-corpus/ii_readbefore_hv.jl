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
    __cse_0 = x[1]
    __cse_6 = __cse_0 * xd[1]
    sd = __cse_6 + __cse_6
    s = __cse_0 * __cse_0
    __cse_7 = s * sd
    outd[1] = outd[1] + (__cse_7 + __cse_7)
    out[1] = out[1] + s * s
    for i_i = 1:i_n
        __idx_s_stack_0 = (i_i - 1) + 1
        s_stack_d[__idx_s_stack_0] = sd
        s_stack[__idx_s_stack_0] = s
        sd = 0.0
        s = 0.0
        for i_j = 1:i_m
            __cse_8 = x[i_i]
            __cse_9 = x[i_j]
            sd = sd + (__cse_8 * xd[i_j] + __cse_9 * xd[i_i])
            s = s + __cse_9 * __cse_8
        end
        __cse_10 = s * sd
        outd[i_i] = outd[i_i] + (__cse_10 + __cse_10)
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_5 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        s_stack_d[__idx_s_stack_5] = sd
        s_stack[__idx_s_stack_5] = s
    end
    __idx_s_stack_3 = ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1
    s_stack_d[__idx_s_stack_3] = sd
    s_stack[__idx_s_stack_3] = s
    __idx_s_stack_0 = ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_i = i_n:-1:1
        __idx_s_stack_0 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __cse_11 = outb[i_i]
        __cse_1d = __cse_11 * sd + s * outbd[i_i]
        __cse_1 = s * __cse_11
        sbd = sbd + __cse_1d
        sb = sb + __cse_1
        sbd = sbd + __cse_1d
        sb = sb + __cse_1
        for i_j = 1:i_m
            __cse_2d = xd[i_j]
            __cse_2 = x[i_j]
            __cse_3d = xd[i_i]
            __cse_3 = x[i_i]
            sd = sd + (__cse_3 * __cse_2d + __cse_2 * __cse_3d)
            s = s + __cse_2 * __cse_3
            xbd[i_j] = xbd[i_j] + (sb * __cse_3d + __cse_3 * sbd)
            xb[i_j] = xb[i_j] + __cse_3 * sb
            xbd[i_i] = xbd[i_i] + (sb * __cse_2d + __cse_2 * sbd)
            xb[i_i] = xb[i_i] + __cse_2 * sb
        end
        __idx_s_stack_0 = (i_i - 1) + 1
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        sbd = 0.0
        sb = 0.0
    end
    __cse_12 = outb[1]
    __cse_4d = __cse_12 * sd + s * outbd[1]
    __cse_4 = s * __cse_12
    sbd = sbd + __cse_4d
    sb = sb + __cse_4
    sbd = sbd + __cse_4d
    sb = sb + __cse_4
    __cse_13 = x[1]
    __cse_5d = sb * xd[1] + __cse_13 * sbd
    __cse_5 = __cse_13 * sb
    xbd[1] = xbd[1] + __cse_5d
    xb[1] = xb[1] + __cse_5
    xbd[1] = xbd[1] + __cse_5d
    xb[1] = xb[1] + __cse_5
    sbd = 0.0
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
