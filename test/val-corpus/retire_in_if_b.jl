function initstacks_retire_in_if_b(levels, n)
    cur = n
    prefix_branch_stack_1 = Vector{Int}(undef, max(0, div(levels - 1, 1) + 1))
    __tot_branch_stack_1 = 0
    prefix_t_stack_1 = Vector{Int}(undef, max(0, div(levels - 1, 1) + 1))
    __tot_t_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(levels - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    val_cur_1 = Vector{Int64}(undef, max(0, div(levels - 1, 1) + 1))
    for i_l = 1:levels
        prefix_branch_stack_1[(i_l - 1) + 1] = __tot_branch_stack_1
        prefix_t_stack_1[(i_l - 1) + 1] = __tot_t_stack_1
        prefix_tripcount_stack_1[(i_l - 1) + 1] = __tot_tripcount_stack_1
        val_cur_1[(i_l - 1) + 1] = cur
        __tot_branch_stack_1 = __tot_branch_stack_1 + (1 + max(0, div(cur - 1, 1) + 1))
        __tot_t_stack_1 = __tot_t_stack_1 + (((max(0, div(cur - 1, 1) + 1) + max(0, div(cur - 1, 1) + 1)) + 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
        if n >= 0
            cur = div(cur + 1, 2)
        end
    end
    branch_stack = Vector{Int64}(undef, __tot_branch_stack_1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    t_stack = Vector{Float64}(undef, __tot_t_stack_1 + 1)
    return (branch_stack, tripcount_stack, t_stack, prefix_branch_stack_1, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
end

function retire_in_if_b(x, xb, y, yb, n, levels, out, outb, branch_stack, tripcount_stack, t_stack, prefix_branch_stack_1, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
    t = 0.0
    tb = 0.0
    cur = n
    for i_l = 1:levels
        if n >= 0
            __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_l - 1) + 1] + 1
            branch_stack[__idx_branch_stack_1_0] = 1
            __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_l - 1) + 1] + 1
            tripcount_stack[__idx_tripcount_stack_1_0] = cur
            for i = 1:cur
                if n >= 0
                    __icse_0 = (i - 1) + 1
                    __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_l - 1) + 1] + 1) + __icse_0
                    branch_stack[__idx_branch_stack_1_0] = 1
                    __idx_t_stack_1_0 = prefix_t_stack_1[(i_l - 1) + 1] + __icse_0
                    t_stack[__idx_t_stack_1_0] = t
                    __cse_1 = x[i]
                    t = __cse_1 * __cse_1
                    y[i] = y[i] + t * t
                else
                    __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_l - 1) + 1] + 1) + ((i - 1) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                __idx_t_stack_1_1 = (prefix_t_stack_1[(i_l - 1) + 1] + max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)) + ((i - 1) + 1)
                t_stack[__idx_t_stack_1_1] = t
            end
            cur = div(cur + 1, 2)
            __icse_2 = max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)
            __idx_t_stack_1_4 = (prefix_t_stack_1[(i_l - 1) + 1] + (__icse_2 + __icse_2)) + 1
            t_stack[__idx_t_stack_1_4] = t
        else
            __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_l - 1) + 1] + 1
            branch_stack[__idx_branch_stack_1_0] = 0
        end
        __icse_3 = max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)
        __idx_t_stack_1_1 = (prefix_t_stack_1[(i_l - 1) + 1] + ((__icse_3 + __icse_3) + 1)) + 1
        t_stack[__idx_t_stack_1_1] = t
    end
    out[1] = y[1]
    __icse_4 = __tot_t_stack_1 + 1
    __idx_t_stack_3 = __icse_4
    t_stack[__idx_t_stack_3] = t
    __idx_t_stack_0 = __icse_4
    t = t_stack[__idx_t_stack_0]
    __oldb_0 = outb[1]
    outb[1] = 0.0
    yb[1] = yb[1] + __oldb_0
    for i_l = levels:-1:1
        __icse_5 = max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)
        __idx_t_stack_1_0 = (prefix_t_stack_1[(i_l - 1) + 1] + ((__icse_5 + __icse_5) + 1)) + 1
        t = t_stack[__idx_t_stack_1_0]
        __idx_branch_stack_1_2 = prefix_branch_stack_1[(i_l - 1) + 1] + 1
        __branch = branch_stack[__idx_branch_stack_1_2]
        if __branch == 1
            __icse_6 = max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)
            __idx_t_stack_1_0 = (prefix_t_stack_1[(i_l - 1) + 1] + (__icse_6 + __icse_6)) + 1
            t = t_stack[__idx_t_stack_1_0]
            __idx_tripcount_stack_1_2 = prefix_tripcount_stack_1[(i_l - 1) + 1] + 1
            cur = tripcount_stack[__idx_tripcount_stack_1_2]
            for i = cur:-1:1
                __icse_7 = (i - 1) + 1
                __idx_t_stack_1_0 = (prefix_t_stack_1[(i_l - 1) + 1] + max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)) + __icse_7
                t = t_stack[__idx_t_stack_1_0]
                __idx_branch_stack_1_2 = (prefix_branch_stack_1[(i_l - 1) + 1] + 1) + __icse_7
                __branch = branch_stack[__idx_branch_stack_1_2]
                if __branch == 1
                    __cse_8 = t * yb[i]
                    tb = tb + __cse_8
                    tb = tb + __cse_8
                    __idx_t_stack_1_0 = prefix_t_stack_1[(i_l - 1) + 1] + ((i - 1) + 1)
                    t = t_stack[__idx_t_stack_1_0]
                    __oldb_2 = tb
                    tb = 0.0
                    __cse_9 = x[i] * __oldb_2
                    xb[i] = xb[i] + __cse_9
                    xb[i] = xb[i] + __cse_9
                end
            end
        end
    end
    return nothing
end

function retire_in_if(x, y, n, levels, out)
    cur = n
    for i_l = 1:levels
        if n >= 0
            for i = 1:cur
                if n >= 0
                    t = x[i] * x[i]
                    y[i] = y[i] + t * t
                end
            end
            cur = div(cur + 1, 2)
        end
    end
    out[1] = y[1]
end
