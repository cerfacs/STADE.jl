function initstacks_entry_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return (branch_stack, s_stack)
end

function entry_branch_hv(x, xb, y, yb, flag, flagb, i_n, out, outb, xd, xbd, yd, ybd, flagd, flagbd, outd, outbd, branch_stack, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    __cse_1 = y[1]
    __cse_2 = x[1]
    sd = __cse_1 * xd[1] + __cse_2 * yd[1]
    s = __cse_2 * __cse_1
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            __idx_branch_stack_0 = (i_i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            __idx_s_stack_0 = (i_i - 1) + 1
            s_stack_d[__idx_s_stack_0] = sd
            s_stack[__idx_s_stack_0] = s
            __cse_3 = y[i_i]
            __cse_4 = x[i_i]
            sd = __cse_3 * xd[i_i] + __cse_4 * yd[i_i]
            s = __cse_4 * __cse_3
        else
            __idx_branch_stack_0 = (i_i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
        __cse_5 = s * sd
        outd[i_i] = outd[i_i] + (__cse_5 + __cse_5)
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_2 = max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)
        s_stack_d[__idx_s_stack_2] = sd
        s_stack[__idx_s_stack_2] = s
    end
    __idx_s_stack_2 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1
    s_stack_d[__idx_s_stack_2] = sd
    s_stack[__idx_s_stack_2] = s
    __idx_s_stack_0 = (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_i = i_n:-1:1
        __idx_s_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __cse_6 = outb[i_i]
        __cse_0d = __cse_6 * sd + s * outbd[i_i]
        __cse_0 = s * __cse_6
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        __idx_branch_stack_4 = (i_i - 1) + 1
        __branch = branch_stack[__idx_branch_stack_4]
        if __branch == 1
            __idx_s_stack_0 = (i_i - 1) + 1
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            __cse_7 = y[i_i]
            xbd[i_i] = xbd[i_i] + (sb * yd[i_i] + __cse_7 * sbd)
            xb[i_i] = xb[i_i] + __cse_7 * sb
            __cse_8 = x[i_i]
            ybd[i_i] = ybd[i_i] + (sb * xd[i_i] + __cse_8 * sbd)
            yb[i_i] = yb[i_i] + __cse_8 * sb
            sbd = 0.0
            sb = 0.0
        end
    end
    __cse_9 = y[1]
    xbd[1] = xbd[1] + (sb * yd[1] + __cse_9 * sbd)
    xb[1] = xb[1] + __cse_9 * sb
    __cse_10 = x[1]
    ybd[1] = ybd[1] + (sb * xd[1] + __cse_10 * sbd)
    yb[1] = yb[1] + __cse_10 * sb
    sbd = 0.0
    sb = 0.0
    return nothing
end

function entry_branch(x, y, flag, i_n, out)
    s = x[1] * y[1]
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            s = x[i_i] * y[i_i]
        end
        out[i_i] = out[i_i] + s * s
    end
end
