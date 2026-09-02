function initstacks_cse_zerotrip_b(i_npass, i_w0)
    width = i_w0
    prefix_s_stack_1 = Vector{Int}(undef, max(0, div(i_npass - 1, 1) + 1))
    __tot_s_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(i_npass - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    for i_p = 1:i_npass
        prefix_s_stack_1[(i_p - 1) + 1] = __tot_s_stack_1
        prefix_tripcount_stack_1[(i_p - 1) + 1] = __tot_tripcount_stack_1
        width = width - 3
        __tot_s_stack_1 = __tot_s_stack_1 + 1
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    s_stack = Vector{Float64}(undef, __tot_s_stack_1 + 1)
    return (tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
end

function cse_zerotrip_b(out, outb, u, ub, w, wb, i_npass, i_w0, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s = 0.0
    sb = 0.0
    s = u[1] * w[1]
    width = i_w0
    for i_p = 1:i_npass
        __cse_0 = u[1]
        s = s + __cse_0 * w[1] * __cse_0
        __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_1] = width
        for i_j = 1:width
            s = s + u[i_j] * w[i_j]
        end
        out[i_p] = s * (u[1] * w[1])
        width = width - 3
        __idx_s_stack_1_6 = prefix_s_stack_1[(i_p - 1) + 1] + 1
        s_stack[__idx_s_stack_1_6] = s
    end
    __icse_1 = __tot_s_stack_1 + 1
    __idx_s_stack_3 = __icse_1
    s_stack[__idx_s_stack_3] = s
    __idx_s_stack_0 = __icse_1
    s = s_stack[__idx_s_stack_0]
    for i_p = i_npass:-1:1
        __idx_s_stack_1_0 = prefix_s_stack_1[(i_p - 1) + 1] + 1
        s = s_stack[__idx_s_stack_1_0]
        __oldb_0 = outb[i_p]
        outb[i_p] = 0.0
        __cse_2 = u[1]
        __cse_3 = w[1]
        sb = sb + (__cse_2 * __cse_3) * __oldb_0
        __cse_4 = s * __oldb_0
        ub[1] = ub[1] + __cse_3 * __cse_4
        wb[1] = wb[1] + __cse_2 * __cse_4
        __idx_tripcount_stack_1_7 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        width = tripcount_stack[__idx_tripcount_stack_1_7]
        for i_j = width:-1:1
            ub[i_j] = ub[i_j] + w[i_j] * sb
            wb[i_j] = wb[i_j] + u[i_j] * sb
        end
        __cse_5 = w[1]
        __cse_6 = u[1]
        ub[1] = ub[1] + (__cse_5 * __cse_6) * sb
        wb[1] = wb[1] + (__cse_6 * __cse_6) * sb
        ub[1] = ub[1] + (__cse_6 * __cse_5) * sb
    end
    __oldb_0 = sb
    sb = 0.0
    ub[1] = ub[1] + w[1] * __oldb_0
    wb[1] = wb[1] + u[1] * __oldb_0
    return nothing
end

function cse_zerotrip(out, u, w, i_npass, i_w0)
    s = u[1] * w[1]
    width = i_w0
    for i_p = 1:i_npass
        s = s + u[1] * w[1] * u[1]
        for i_j = 1:width
            s = s + u[i_j] * w[i_j]
        end
        out[i_p] = s * (u[1] * w[1])
        width = width - 3
    end
end
