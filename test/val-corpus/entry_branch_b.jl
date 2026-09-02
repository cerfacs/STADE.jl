function initstacks_entry_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, (max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + 1)
    return (branch_stack, s_stack)
end

function entry_branch_b(x, xb, y, yb, flag, flagb, i_n, out, outb, branch_stack, s_stack)
    s = 0.0
    sb = 0.0
    s = x[1] * y[1]
    for i_i = 1:i_n
        if flag[i_i] > 0.0
            __icse_0 = (i_i - 1) + 1
            __idx_branch_stack_0 = __icse_0
            branch_stack[__idx_branch_stack_0] = 1
            __idx_s_stack_0 = __icse_0
            s_stack[__idx_s_stack_0] = s
            s = x[i_i] * y[i_i]
        else
            __idx_branch_stack_0 = (i_i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
        out[i_i] = out[i_i] + s * s
        __idx_s_stack_2 = max(0, div(i_n - 1, 1) + 1) + ((i_i - 1) + 1)
        s_stack[__idx_s_stack_2] = s
    end
    __icse_1 = max(0, div(i_n - 1, 1) + 1)
    __icse_2 = (__icse_1 + __icse_1) + 1
    __idx_s_stack_2 = __icse_2
    s_stack[__idx_s_stack_2] = s
    __idx_s_stack_0 = __icse_2
    s = s_stack[__idx_s_stack_0]
    for i_i = i_n:-1:1
        __icse_3 = (i_i - 1) + 1
        __idx_s_stack_0 = max(0, div(i_n - 1, 1) + 1) + __icse_3
        s = s_stack[__idx_s_stack_0]
        __cse_4 = s * outb[i_i]
        sb = sb + __cse_4
        sb = sb + __cse_4
        __idx_branch_stack_4 = __icse_3
        __branch = branch_stack[__idx_branch_stack_4]
        if __branch == 1
            __idx_s_stack_0 = (i_i - 1) + 1
            s = s_stack[__idx_s_stack_0]
            __oldb_2 = sb
            sb = 0.0
            xb[i_i] = xb[i_i] + y[i_i] * __oldb_2
            yb[i_i] = yb[i_i] + x[i_i] * __oldb_2
        end
    end
    __oldb_0 = sb
    sb = 0.0
    xb[1] = xb[1] + y[1] * __oldb_0
    yb[1] = yb[1] + x[1] * __oldb_0
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
