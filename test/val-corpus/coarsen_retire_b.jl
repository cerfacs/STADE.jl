function initstacks_coarsen_retire_b(levels, n)
    cur = n
    prefix_t_stack_1 = Vector{Int}(undef, max(0, div(levels - 1, 1) + 1))
    __tot_t_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(levels - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    for i_l = 1:levels
        prefix_t_stack_1[(i_l - 1) + 1] = __tot_t_stack_1
        prefix_tripcount_stack_1[(i_l - 1) + 1] = __tot_tripcount_stack_1
        cur = div(cur + 1, 2)
        __tot_t_stack_1 = __tot_t_stack_1 + 1
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    return (tripcount_stack, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1)
end

function coarsen_retire_b(x, xb, y, yb, n, levels, out, outb, tripcount_stack, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1)
    t = 0.0
    tb = 0.0
    cur = n
    for i_l = 1:levels
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_l - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_0] = cur
        for i = 1:cur
            __cse_0 = x[i]
            t = __cse_0 * __cse_0
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
    __oldb_0 = outb[1]
    outb[1] = 0.0
    yb[1] = yb[1] + __oldb_0
    for i_l = levels:-1:1
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_l - 1) + 1] + 1
        cur = tripcount_stack[__idx_tripcount_stack_1_0]
        for i = cur:-1:1
            __cse_1 = x[i]
            t = __cse_1 * __cse_1
            __cse_2 = t * yb[i]
            tb = tb + __cse_2
            tb = tb + __cse_2
            __oldb_0 = tb
            tb = 0.0
            __cse_3 = __cse_1 * __oldb_0
            xb[i] = xb[i] + __cse_3
            xb[i] = xb[i] + __cse_3
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
