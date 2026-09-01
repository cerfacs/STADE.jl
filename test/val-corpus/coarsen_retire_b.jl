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

function coarsen_retire_b(x, xb, y, yb, n, levels, out, outb, tripcount_stack, t_stack, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
    t = 0.0
    tb = 0.0
    cur = n
    for i_l = 1:levels
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_l - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_0] = cur
        for i = 1:cur
            __idx_t_stack_1_0 = prefix_t_stack_1[(i_l - 1) + 1] + ((i - 1) + 1)
            t_stack[__idx_t_stack_1_0] = t
            __cse_0 = x[i]
            t = __cse_0 * __cse_0
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
        __idx_t_stack_1_4 = (prefix_t_stack_1[(i_l - 1) + 1] + max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)) + 1
        t_stack[__idx_t_stack_1_4] = t
    end
    out[1] = y[1]
    __idx_t_stack_3 = __tot_t_stack_1 + 1
    t_stack[__idx_t_stack_3] = t
    __idx_t_stack_0 = __tot_t_stack_1 + 1
    t = t_stack[__idx_t_stack_0]
    yb[1] = yb[1] + outb[1]
    outb[1] = 0.0
    for i_l = levels:-1:1
        __idx_t_stack_1_0 = (prefix_t_stack_1[(i_l - 1) + 1] + max(0, div(val_cur_1[(i_l - 1) + 1] - 1, 1) + 1)) + 1
        t = t_stack[__idx_t_stack_1_0]
        __idx_tripcount_stack_1_2 = prefix_tripcount_stack_1[(i_l - 1) + 1] + 1
        cur = tripcount_stack[__idx_tripcount_stack_1_2]
        for i = cur:-1:1
            __cse_1 = t * yb[i]
            tb = tb + __cse_1
            tb = tb + __cse_1
            __idx_t_stack_1_0 = prefix_t_stack_1[(i_l - 1) + 1] + ((i - 1) + 1)
            t = t_stack[__idx_t_stack_1_0]
            __cse_2 = x[i] * tb
            xb[i] = xb[i] + __cse_2
            xb[i] = xb[i] + __cse_2
            tb = 0.0
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
