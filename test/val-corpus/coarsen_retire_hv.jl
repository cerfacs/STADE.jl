function initstacks_coarsen_retire_b(levels, n)
    cur = n
    prefix_t_stack_1 = Vector{Int}(undef, max(0, div(levels - 1, 1) + 1))
    __tot_t_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(levels - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    val_cur_1 = Vector{Int64}(undef, max(0, div(levels - 1, 1) + 1))
    for i_l = 1:levels
        prefix_t_stack_1[(i_l - 1) + 1] = __tot_t_stack_1
        prefix_tripcount_stack_1[(i_l - 1) + 1] = __tot_tripcount_stack_1
        val_cur_1[(i_l - 1) + 1] = cur
        __tot_t_stack_1 = __tot_t_stack_1 + (max(0, div(cur - 1, 1) + 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
        cur = div(cur + 1, 2)
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    t_stack = Vector{Float64}(undef, __tot_t_stack_1 + 1)
    return (tripcount_stack, t_stack, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
end

function coarsen_retire_hv(x, xb, y, yb, n, levels, out, outb, xd, xbd, yd, ybd, outd, outbd, tripcount_stack, t_stack, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
    t_stack_d = Vector{Float64}(undef, length(t_stack))
    t = 0.0
    tb = 0.0
    td = 0.0
    tbd = 0.0
    cur = n
    for i_l = 1:levels
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_l - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_0] = cur
        for i = 1:cur
            __idx_t_stack_1_0 = prefix_t_stack_1[(i_l - 1) + 1] + ((i - 1) + 1)
            t_stack_d[__idx_t_stack_1_0] = td
            t_stack[__idx_t_stack_1_0] = t
            __cse_0d = xd[i]
            __cse_0 = x[i]
            __hcse_0 = __cse_0 * __cse_0d
            td = __hcse_0 + __hcse_0
            t = __cse_0 * __cse_0
            __hcse_1 = t * td
            yd[i] = yd[i] + (__hcse_1 + __hcse_1)
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
        __idx_t_stack_1_4 = (prefix_t_stack_1[(i_l - 1) + 1] + max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)) + 1
        t_stack_d[__idx_t_stack_1_4] = td
        t_stack[__idx_t_stack_1_4] = t
    end
    outd[1] = yd[1]
    out[1] = y[1]
    __ihcse_2 = __tot_t_stack_1 + 1
    __idx_t_stack_3 = __ihcse_2
    t_stack_d[__idx_t_stack_3] = td
    t_stack[__idx_t_stack_3] = t
    __idx_t_stack_0 = __ihcse_2
    td = t_stack_d[__idx_t_stack_0]
    t = t_stack[__idx_t_stack_0]
    __oldb_0d = outbd[1]
    __oldb_0 = outb[1]
    outbd[1] = 0.0
    outb[1] = 0.0
    ybd[1] = ybd[1] + __oldb_0d
    yb[1] = yb[1] + __oldb_0
    for i_l = levels:-1:1
        __idx_t_stack_1_0 = (prefix_t_stack_1[(i_l - 1) + 1] + max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)) + 1
        td = t_stack_d[__idx_t_stack_1_0]
        t = t_stack[__idx_t_stack_1_0]
        __idx_tripcount_stack_1_2 = prefix_tripcount_stack_1[(i_l - 1) + 1] + 1
        cur = tripcount_stack[__idx_tripcount_stack_1_2]
        for i = cur:-1:1
            __hcse_3 = yb[i]
            __cse_1d = __hcse_3 * td + t * ybd[i]
            __cse_1 = t * __hcse_3
            tbd = tbd + __cse_1d
            tb = tb + __cse_1
            tbd = tbd + __cse_1d
            tb = tb + __cse_1
            __idx_t_stack_1_0 = prefix_t_stack_1[(i_l - 1) + 1] + ((i - 1) + 1)
            td = t_stack_d[__idx_t_stack_1_0]
            t = t_stack[__idx_t_stack_1_0]
            __oldb_2d = tbd
            __oldb_2 = tb
            tbd = 0.0
            tb = 0.0
            __hcse_4 = x[i]
            __cse_2d = __oldb_2 * xd[i] + __hcse_4 * __oldb_2d
            __cse_2 = __hcse_4 * __oldb_2
            xbd[i] = xbd[i] + __cse_2d
            xb[i] = xb[i] + __cse_2
            xbd[i] = xbd[i] + __cse_2d
            xb[i] = xb[i] + __cse_2
        end
    end
    return nothing
end

function coarsen_retire(x, y, n, levels, out)
    cur = n
    for i_l = 1:levels
        for i = 1:cur
            t = x[i] * x[i]
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
end
