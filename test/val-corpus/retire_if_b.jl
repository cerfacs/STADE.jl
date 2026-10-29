function initstacks_retire_if_b(i_n, i_nlev, i_nrep)
    cur = i_n
    prefix_branch_stack_1 = Vector{Int}(undef, max(0, div(i_nlev - 1, 1) + 1))
    __tot_branch_stack_1 = 0
    prefix_t_stack_1 = Vector{Int}(undef, max(0, div(i_nlev - 1, 1) + 1))
    __tot_t_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(i_nlev - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    val_cur_1 = Vector{Int64}(undef, max(0, div(i_nlev - 1, 1) + 1))
    for i_l = 1:i_nlev
        prefix_branch_stack_1[(i_l - 1) + 1] = __tot_branch_stack_1
        prefix_t_stack_1[(i_l - 1) + 1] = __tot_t_stack_1
        prefix_tripcount_stack_1[(i_l - 1) + 1] = __tot_tripcount_stack_1
        val_cur_1[(i_l - 1) + 1] = cur
        __tot_branch_stack_1 = __tot_branch_stack_1 + max(0, div(i_nrep - 1, 1) + 1)
        __tot_t_stack_1 = __tot_t_stack_1 + (((max(0, div(i_nrep - 1, 1) + 1) * max(0, div(cur - 1, 1) + 1) + max(0, div(i_nrep - 1, 1) + 1)) + max(0, div(i_nrep - 1, 1) + 1)) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + (max(0, div(i_nrep - 1, 1) + 1) + max(0, div(i_nrep - 1, 1) + 1))
        cur = div(cur + 1, 2)
    end
    branch_stack = Vector{Int64}(undef, __tot_branch_stack_1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    t_stack = Vector{Float64}(undef, __tot_t_stack_1 + 1)
    return (branch_stack, tripcount_stack, t_stack, prefix_branch_stack_1, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
end

function retire_if_b(x, xb, y, yb, i_n, i_nlev, i_nrep, out, outb, branch_stack, tripcount_stack, t_stack, prefix_branch_stack_1, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
    t = 0.0
    tb = 0.0
    cur = i_n
    for i_l = 1:i_nlev
        for i_r = 1:i_nrep
            if i_r > 1
                __icse_0 = (i_r - 1) + 1
                __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_l - 1) + 1] + __icse_0
                branch_stack[__idx_branch_stack_1_0] = 1
                __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_l - 1) + 1] + __icse_0
                tripcount_stack[__idx_tripcount_stack_1_0] = cur
                for i = 1:cur
                    __idx_t_stack_1_0 = prefix_t_stack_1[(i_l - 1) + 1] + (((i_r - 1) * (div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1) + (i - 1)) + 1)
                    t_stack[__idx_t_stack_1_0] = t
                    __cse_1 = x[i]
                    t = __cse_1 * __cse_1
                    y[i] = y[i] + t * t
                end
                __idx_t_stack_1_3 = (prefix_t_stack_1[(i_l - 1) + 1] + max(0, div(i_nrep - 1, 1) + 1) * max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)) + ((i_r - 1) + 1)
                t_stack[__idx_t_stack_1_3] = t
            else
                __icse_2 = (i_r - 1) + 1
                __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_l - 1) + 1] + __icse_2
                branch_stack[__idx_branch_stack_1_0] = 0
                __idx_tripcount_stack_1_0 = (prefix_tripcount_stack_1[(i_l - 1) + 1] + max(0, div(i_nrep - 1, 1) + 1)) + __icse_2
                tripcount_stack[__idx_tripcount_stack_1_0] = cur
                for i = 1:cur
                    y[i] = y[i] + x[i]
                end
            end
            __icse_3 = max(0, div(i_nrep - 1, 1) + 1)
            __idx_t_stack_1_1 = (prefix_t_stack_1[(i_l - 1) + 1] + (__icse_3 * max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1) + __icse_3)) + ((i_r - 1) + 1)
            t_stack[__idx_t_stack_1_1] = t
        end
        cur = div(cur + 1, 2)
        __icse_4 = max(0, div(i_nrep - 1, 1) + 1)
        __idx_t_stack_1_2 = (prefix_t_stack_1[(i_l - 1) + 1] + ((__icse_4 * max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1) + __icse_4) + __icse_4)) + 1
        t_stack[__idx_t_stack_1_2] = t
    end
    out[1] = y[1]
    __icse_5 = __tot_t_stack_1 + 1
    __idx_t_stack_3 = __icse_5
    t_stack[__idx_t_stack_3] = t
    __idx_t_stack_0 = __icse_5
    t = t_stack[__idx_t_stack_0]
    __oldb_0 = outb[1]
    outb[1] = 0.0
    yb[1] = yb[1] + __oldb_0
    for i_l = i_nlev:-1:1
        __icse_6 = max(0, div(i_nrep - 1, 1) + 1)
        __idx_t_stack_1_0 = (prefix_t_stack_1[(i_l - 1) + 1] + ((__icse_6 * max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1) + __icse_6) + __icse_6)) + 1
        t = t_stack[__idx_t_stack_1_0]
        for i_r = i_nrep:-1:1
            __icse_7 = max(0, div(i_nrep - 1, 1) + 1)
            __icse_8 = (i_r - 1) + 1
            __idx_t_stack_1_0 = (prefix_t_stack_1[(i_l - 1) + 1] + (__icse_7 * max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1) + __icse_7)) + __icse_8
            t = t_stack[__idx_t_stack_1_0]
            __idx_branch_stack_1_2 = prefix_branch_stack_1[(i_l - 1) + 1] + __icse_8
            __branch = branch_stack[__idx_branch_stack_1_2]
            if __branch == 1
                __icse_9 = (i_r - 1) + 1
                __idx_t_stack_1_0 = (prefix_t_stack_1[(i_l - 1) + 1] + max(0, div(i_nrep - 1, 1) + 1) * max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)) + __icse_9
                t = t_stack[__idx_t_stack_1_0]
                __idx_tripcount_stack_1_2 = prefix_tripcount_stack_1[(i_l - 1) + 1] + __icse_9
                cur = tripcount_stack[__idx_tripcount_stack_1_2]
                for i = cur:-1:1
                    __cse_10 = t * yb[i]
                    tb = tb + __cse_10
                    tb = tb + __cse_10
                    __idx_t_stack_1_0 = prefix_t_stack_1[(i_l - 1) + 1] + (((i_r - 1) * (div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1) + (i - 1)) + 1)
                    t = t_stack[__idx_t_stack_1_0]
                    __oldb_2 = tb
                    tb = 0.0
                    __cse_11 = x[i] * __oldb_2
                    xb[i] = xb[i] + __cse_11
                    xb[i] = xb[i] + __cse_11
                end
            else
                __idx_tripcount_stack_1_0 = (prefix_tripcount_stack_1[(i_l - 1) + 1] + max(0, div(i_nrep - 1, 1) + 1)) + ((i_r - 1) + 1)
                cur = tripcount_stack[__idx_tripcount_stack_1_0]
                for i = cur:-1:1
                    xb[i] = xb[i] + yb[i]
                end
            end
        end
    end
    return nothing
end

function retire_if(x, y, i_n, i_nlev, i_nrep, out)
    cur = i_n
    for i_l = 1:i_nlev
        for i_r = 1:i_nrep
            if i_r > 1
                for i = 1:cur
                    t = x[i] * x[i]
                    y[i] = y[i] + t * t
                end
            else
                for i = 1:cur
                    y[i] = y[i] + x[i]
                end
            end
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
end
