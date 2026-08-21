function initstacks_coarsen_retire_b(levels, n)
    cur = n
    prefix_t_stack_1 = Vector{Int}(undef, div(levels - 1, 1) + 1)
    __tot_t_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, div(levels - 1, 1) + 1)
    __tot_tripcount_stack_1 = 0
    val_cur_1 = Vector{Int64}(undef, div(levels - 1, 1) + 1)
    for i_seq_l = 1:levels
        prefix_t_stack_1[(i_seq_l - 1) + 1] = __tot_t_stack_1
        prefix_tripcount_stack_1[(i_seq_l - 1) + 1] = __tot_tripcount_stack_1
        cur = div(cur + 1, 2)
        val_cur_1[(i_seq_l - 1) + 1] = cur
        __tot_t_stack_1 = __tot_t_stack_1 + ((div(cur - 1, 1) + 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    t_stack = Vector{Float64}(undef, __tot_t_stack_1 + 1)
    return (tripcount_stack, t_stack, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
end

function coarsen_retire_b(x, xb, y, yb, n, levels, out, outb, tripcount_stack, t_stack, prefix_t_stack_1, prefix_tripcount_stack_1, __tot_t_stack_1, __tot_tripcount_stack_1, val_cur_1)
    t = 0.0
    tb = 0.0
    cur = n
    for i_seq_l = 1:levels
        tripcount_stack[prefix_tripcount_stack_1[(i_seq_l - 1) + 1] + 1] = cur
        for i = 1:cur
            t = x[i] * x[i]
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
    yb[1] = yb[1] + outb[1]
    outb[1] = 0.0
    for i_seq_l = levels:-1:1
        for i = 1:cur
            t = x[i] * x[i]
            tb = tb + t * yb[i]
            tb = tb + t * yb[i]
            xb[i] = xb[i] + x[i] * tb
            xb[i] = xb[i] + x[i] * tb
            tb = 0.0
        end
    end
    return nothing
end

function coarsen_retire(x, y, n, levels, out)
    cur = n
    for i_seq_l = 1:levels
        for i = 1:cur
            t = x[i] * x[i]
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
    end
    out[1] = y[1]
end
