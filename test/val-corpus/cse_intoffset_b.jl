function initstacks_cse_intoffset_b(i_base, i_nrow)
    prefix_s_stack_1 = Vector{Int}(undef, max(0, div(i_nrow - 1, 1) + 1))
    __tot_s_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(i_nrow - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    for i_r = 1:i_nrow
        prefix_s_stack_1[(i_r - 1) + 1] = __tot_s_stack_1
        prefix_tripcount_stack_1[(i_r - 1) + 1] = __tot_tripcount_stack_1
        i_len = i_base + i_r
        i_off = (i_r - 1) * i_base + i_len
        i_lo = ((i_r - 1) * i_base + i_len) - i_len
        i_hi = (i_r - 1) * i_base + i_len + 1
        s = 0.0
        __tot_s_stack_1 = __tot_s_stack_1 + 1
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    s_stack = Vector{Float64}(undef, __tot_s_stack_1)
    return (tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
end

function cse_intoffset_b(out, outb, u, ub, i_nrow, i_base, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s = 0.0
    sb = 0.0
    for i_r = 1:i_nrow
        i_len = i_base + i_r
        __icse_0 = (i_r - 1) * i_base
        __icse_1 = __icse_0 + i_len
        i_off = __icse_1
        i_lo = __icse_1 - i_len
        i_hi = __icse_0 + i_len + 1
        s = 0.0
        __idx_tripcount_stack_1_5 = prefix_tripcount_stack_1[(i_r - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_5] = i_len
        for i_j = 1:i_len
            s = s + u[(i_r - 1) * i_base + i_j] * u[i_lo + i_j]
        end
        out[i_r] = s * u[i_off] + s * u[i_hi]
        __idx_s_stack_1_9 = prefix_s_stack_1[(i_r - 1) + 1] + 1
        s_stack[__idx_s_stack_1_9] = s
    end
    for i_r = i_nrow:-1:1
        __idx_s_stack_1_0 = prefix_s_stack_1[(i_r - 1) + 1] + 1
        s = s_stack[__idx_s_stack_1_0]
        i_len = i_base + i_r
        __icse_2 = (i_r - 1) * i_base
        __icse_3 = __icse_2 + i_len
        i_off = __icse_3
        i_lo = __icse_3 - i_len
        i_hi = __icse_2 + i_len + 1
        __oldb_0 = outb[i_r]
        outb[i_r] = 0.0
        sb = sb + u[i_off] * __oldb_0
        __cse_4 = s * __oldb_0
        ub[i_off] = ub[i_off] + __cse_4
        sb = sb + u[i_hi] * __oldb_0
        ub[i_hi] = ub[i_hi] + __cse_4
        __idx_tripcount_stack_1_12 = prefix_tripcount_stack_1[(i_r - 1) + 1] + 1
        i_len = tripcount_stack[__idx_tripcount_stack_1_12]
        for i_j = i_len:-1:1
            ub[(i_r - 1) * i_base + i_j] = ub[(i_r - 1) * i_base + i_j] + u[i_lo + i_j] * sb
            ub[i_lo + i_j] = ub[i_lo + i_j] + u[(i_r - 1) * i_base + i_j] * sb
        end
        sb = 0.0
    end
    return nothing
end

function cse_intoffset(out, u, i_nrow, i_base)
    for i_r = 1:i_nrow
        i_len = i_base + i_r
        i_off = (i_r - 1) * i_base + i_len
        i_lo = ((i_r - 1) * i_base + i_len) - i_len
        i_hi = (i_r - 1) * i_base + i_len + 1
        s = 0.0
        for i_j = 1:i_len
            s = s + u[(i_r - 1) * i_base + i_j] * u[i_lo + i_j]
        end
        out[i_r] = s * u[i_off] + s * u[i_hi]
    end
end
