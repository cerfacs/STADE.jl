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
        val_cur_1[(i_seq_l - 1) + 1] = cur
        __tot_t_stack_1 = __tot_t_stack_1 + ((div(cur - 1, 1) + 1) + 1)
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
    for i_seq_l = 1:levels
        tripcount_stack[prefix_tripcount_stack_1[(i_seq_l - 1) + 1] + 1] = cur
        for i = 1:cur
            t_stack_d[prefix_t_stack_1[(i_seq_l - 1) + 1] + ((i - 1) + 1)] = td
            t_stack[prefix_t_stack_1[(i_seq_l - 1) + 1] + ((i - 1) + 1)] = t
            td = x[i] * xd[i] + x[i] * xd[i]
            t = x[i] * x[i]
            yd[i] = yd[i] + (t * td + t * td)
            y[i] = y[i] + t * t
        end
        cur = div(cur + 1, 2)
        t_stack_d[(prefix_t_stack_1[(i_seq_l - 1) + 1] + (div(val_cur_1[(i_seq_l - 1) + 1] - 1, 1) + 1)) + 1] = td
        t_stack[(prefix_t_stack_1[(i_seq_l - 1) + 1] + (div(val_cur_1[(i_seq_l - 1) + 1] - 1, 1) + 1)) + 1] = t
    end
    outd[1] = yd[1]
    out[1] = y[1]
    t_stack_d[__tot_t_stack_1 + 1] = td
    t_stack[__tot_t_stack_1 + 1] = t
    td = t_stack_d[__tot_t_stack_1 + 1]
    t = t_stack[__tot_t_stack_1 + 1]
    ybd[1] = ybd[1] + outbd[1]
    yb[1] = yb[1] + outb[1]
    outbd[1] = 0.0
    outb[1] = 0.0
    for i_seq_l = levels:-1:1
        td = t_stack_d[(prefix_t_stack_1[(i_seq_l - 1) + 1] + (div(val_cur_1[(i_seq_l - 1) + 1] - 1, 1) + 1)) + 1]
        t = t_stack[(prefix_t_stack_1[(i_seq_l - 1) + 1] + (div(val_cur_1[(i_seq_l - 1) + 1] - 1, 1) + 1)) + 1]
        cur = tripcount_stack[prefix_tripcount_stack_1[(i_seq_l - 1) + 1] + 1]
        for i = cur:-1:1
            tbd = tbd + (yb[i] * td + t * ybd[i])
            tb = tb + t * yb[i]
            tbd = tbd + (yb[i] * td + t * ybd[i])
            tb = tb + t * yb[i]
            td = t_stack_d[prefix_t_stack_1[(i_seq_l - 1) + 1] + ((i - 1) + 1)]
            t = t_stack[prefix_t_stack_1[(i_seq_l - 1) + 1] + ((i - 1) + 1)]
            xbd[i] = xbd[i] + (tb * xd[i] + x[i] * tbd)
            xb[i] = xb[i] + x[i] * tb
            xbd[i] = xbd[i] + (tb * xd[i] + x[i] * tbd)
            xb[i] = xb[i] + x[i] * tb
            tbd = 0.0
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
