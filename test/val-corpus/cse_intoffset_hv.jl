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

function cse_intoffset_hv(out, outb, u, ub, i_nrow, i_base, outd, outbd, ud, ubd, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_r = 1:i_nrow
        i_len = i_base + i_r
        __icse_0 = (i_r - 1) * i_base
        __icse_1 = __icse_0 + i_len
        i_off = __icse_1
        i_lo = __icse_1 - i_len
        i_hi = __icse_0 + i_len + 1
        sd = 0.0
        s = 0.0
        __idx_tripcount_stack_1_5 = prefix_tripcount_stack_1[(i_r - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_5] = i_len
        for i_j = 1:i_len
            __hcse_0 = u[i_lo + i_j]
            __hcse_1 = u[(i_r - 1) * i_base + i_j]
            sd = sd + (__hcse_0 * ud[(i_r - 1) * i_base + i_j] + __hcse_1 * ud[i_lo + i_j])
            s = s + __hcse_1 * __hcse_0
        end
        __hcse_2 = u[i_off]
        __hcse_3 = u[i_hi]
        outd[i_r] = (__hcse_2 * sd + s * ud[i_off]) + (__hcse_3 * sd + s * ud[i_hi])
        out[i_r] = s * __hcse_2 + s * __hcse_3
        __idx_s_stack_1_9 = prefix_s_stack_1[(i_r - 1) + 1] + 1
        s_stack_d[__idx_s_stack_1_9] = sd
        s_stack[__idx_s_stack_1_9] = s
    end
    for i_r = i_nrow:-1:1
        __idx_s_stack_1_0 = prefix_s_stack_1[(i_r - 1) + 1] + 1
        sd = s_stack_d[__idx_s_stack_1_0]
        s = s_stack[__idx_s_stack_1_0]
        i_len = i_base + i_r
        __icse_2 = (i_r - 1) * i_base
        __icse_3 = __icse_2 + i_len
        i_off = __icse_3
        i_lo = __icse_3 - i_len
        i_hi = __icse_2 + i_len + 1
        __oldb_0d = outbd[i_r]
        __oldb_0 = outb[i_r]
        outbd[i_r] = 0.0
        outb[i_r] = 0.0
        __hcse_4 = u[i_off]
        sbd = sbd + (__oldb_0 * ud[i_off] + __hcse_4 * __oldb_0d)
        sb = sb + __hcse_4 * __oldb_0
        __cse_4d = __oldb_0 * sd + s * __oldb_0d
        __cse_4 = s * __oldb_0
        ubd[i_off] = ubd[i_off] + __cse_4d
        ub[i_off] = ub[i_off] + __cse_4
        __hcse_5 = u[i_hi]
        sbd = sbd + (__oldb_0 * ud[i_hi] + __hcse_5 * __oldb_0d)
        sb = sb + __hcse_5 * __oldb_0
        ubd[i_hi] = ubd[i_hi] + __cse_4d
        ub[i_hi] = ub[i_hi] + __cse_4
        __idx_tripcount_stack_1_12 = prefix_tripcount_stack_1[(i_r - 1) + 1] + 1
        i_len = tripcount_stack[__idx_tripcount_stack_1_12]
        for i_j = i_len:-1:1
            __hcse_6 = u[i_lo + i_j]
            ubd[(i_r - 1) * i_base + i_j] = ubd[(i_r - 1) * i_base + i_j] + (sb * ud[i_lo + i_j] + __hcse_6 * sbd)
            ub[(i_r - 1) * i_base + i_j] = ub[(i_r - 1) * i_base + i_j] + __hcse_6 * sb
            __hcse_7 = u[(i_r - 1) * i_base + i_j]
            ubd[i_lo + i_j] = ubd[i_lo + i_j] + (sb * ud[(i_r - 1) * i_base + i_j] + __hcse_7 * sbd)
            ub[i_lo + i_j] = ub[i_lo + i_j] + __hcse_7 * sb
        end
        sbd = 0.0
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
