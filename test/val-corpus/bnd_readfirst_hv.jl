function initstacks_bnd_readfirst_b(i_m, i_n)
    s_stack = Vector{Float64}(undef, ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1)
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
        __cse_3 = s * sd
        outd[i_i] = outd[i_i] + (__cse_3 + __cse_3)
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_1 = (i_i - 1) + 1
        s_stack_d[__idx_s_stack_1] = sd
        s_stack[__idx_s_stack_1] = s
        sd = 0.0
        s = 0.0
        for i_j = 1:i_m
            __cse_4 = x[i_j]
            __cse_5 = x[i_i]
            sd = sd + (__cse_4 * xd[i_i] + __cse_5 * xd[i_j])
            s = s + __cse_5 * __cse_4
        end
        __idx_s_stack_5 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        s_stack_d[__idx_s_stack_5] = sd
        s_stack[__idx_s_stack_5] = s
    end
    __idx_s_stack_2 = ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1
    s_stack_d[__idx_s_stack_2] = sd
    s_stack[__idx_s_stack_2] = s
    __idx_s_stack_0 = ((max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_i = i_n:-1:1
        __idx_s_stack_0 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_m - 1, 1) + 1)) + ((i_i - 1) + 1)
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        for i_j = 1:i_m
            __cse_0d = xd[i_i]
            __cse_0 = x[i_i]
            __cse_1d = xd[i_j]
            __cse_1 = x[i_j]
            sd = sd + (__cse_1 * __cse_0d + __cse_0 * __cse_1d)
            s = s + __cse_0 * __cse_1
            xbd[i_i] = xbd[i_i] + (sb * __cse_1d + __cse_1 * sbd)
            xb[i_i] = xb[i_i] + __cse_1 * sb
            xbd[i_j] = xbd[i_j] + (sb * __cse_0d + __cse_0 * sbd)
            xb[i_j] = xb[i_j] + __cse_0 * sb
        end
        __idx_s_stack_0 = (i_i - 1) + 1
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        sbd = 0.0
        sb = 0.0
        __cse_6 = outb[i_i]
        __cse_2d = __cse_6 * sd + s * outbd[i_i]
        __cse_2 = s * __cse_6
        sbd = sbd + __cse_2d
        sb = sb + __cse_2
        sbd = sbd + __cse_2d
        sb = sb + __cse_2
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
