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
    __hcse_0 = y[1]
    __hcse_1 = x[1]
    sd = __hcse_0 * xd[1] + __hcse_1 * yd[1]
    s = __hcse_1 * __hcse_0
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            __icse_0 = (i_i - 1) + 1
            __idx_branch_stack_0 = __icse_0
            branch_stack[__idx_branch_stack_0] = 1
            __idx_s_stack_0 = __icse_0
            s_stack_d[__idx_s_stack_0] = sd
            s_stack[__idx_s_stack_0] = s
            __hcse_2 = y[i_i]
            __hcse_3 = x[i_i]
            sd = __hcse_2 * xd[i_i] + __hcse_3 * yd[i_i]
            s = __hcse_3 * __hcse_2
        else
            __idx_branch_stack_0 = (i_i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
        __hcse_4 = s * sd
        outd[i_i] = outd[i_i] + (__hcse_4 + __hcse_4)
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_2 = max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)
        s_stack_d[__idx_s_stack_2] = sd
        s_stack[__idx_s_stack_2] = s
    end
    __ihcse_5 = max(0, div(i_n - 1, 1) + 1)
    __icse_1 = __ihcse_5
    __idx_s_stack_2 = (__icse_1 + __icse_1) + 1
    s_stack_d[__idx_s_stack_2] = sd
    s_stack[__idx_s_stack_2] = s
    __icse_2 = __ihcse_5
    __idx_s_stack_0 = (__icse_2 + __icse_2) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_i = i_n:-1:1
        __icse_3 = (i_i - 1) + 1
        __idx_s_stack_0 = max(0, div(i_n - 1, 1) + 1) + __icse_3
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __hcse_6 = outb[i_i]
        __cse_4d = __hcse_6 * sd + s * outbd[i_i]
        __cse_4 = s * __hcse_6
        sbd = sbd + __cse_4d
        sb = sb + __cse_4
        sbd = sbd + __cse_4d
        sb = sb + __cse_4
        __idx_branch_stack_4 = __icse_3
        __branch = branch_stack[__idx_branch_stack_4]
        if __branch == 1
            __idx_s_stack_0 = (i_i - 1) + 1
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            __oldb_2d = sbd
            __oldb_2 = sb
            sbd = 0.0
            sb = 0.0
            __hcse_7 = y[i_i]
            xbd[i_i] = xbd[i_i] + (__oldb_2 * yd[i_i] + __hcse_7 * __oldb_2d)
            xb[i_i] = xb[i_i] + __hcse_7 * __oldb_2
            __hcse_8 = x[i_i]
            ybd[i_i] = ybd[i_i] + (__oldb_2 * xd[i_i] + __hcse_8 * __oldb_2d)
            yb[i_i] = yb[i_i] + __hcse_8 * __oldb_2
        end
    end
    __oldb_0d = sbd
    __oldb_0 = sb
    sbd = 0.0
    sb = 0.0
    __hcse_9 = y[1]
    xbd[1] = xbd[1] + (__oldb_0 * yd[1] + __hcse_9 * __oldb_0d)
    xb[1] = xb[1] + __hcse_9 * __oldb_0
    __hcse_10 = x[1]
    ybd[1] = ybd[1] + (__oldb_0 * xd[1] + __hcse_10 * __oldb_0d)
    yb[1] = yb[1] + __hcse_10 * __oldb_0
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
