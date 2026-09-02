function initstacks_entry_empty_b(i_npass, i_w0)
    w = i_w0
    prefix_s_stack_1 = Vector{Int}(undef, max(0, div(i_npass - 1, 1) + 1))
    __tot_s_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(i_npass - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    val_w_1 = Vector{Int64}(undef, max(0, div(i_npass - 1, 1) + 1))
    for i_p = 1:i_npass
        prefix_s_stack_1[(i_p - 1) + 1] = __tot_s_stack_1
        prefix_tripcount_stack_1[(i_p - 1) + 1] = __tot_tripcount_stack_1
        val_w_1[(i_p - 1) + 1] = w
        __tot_s_stack_1 = __tot_s_stack_1 + (max(0, div(w - 1, 1) + 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
        w = w - 3
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    s_stack = Vector{Float64}(undef, __tot_s_stack_1 + 1)
    return (tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1, val_w_1)
end

function entry_empty_hv(x, xb, u, ub, i_npass, i_w0, out, outb, xd, xbd, ud, ubd, outd, outbd, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1, val_w_1)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    __cse_0d = xd[1]
    __cse_0 = x[1]
    __hcse_0 = __cse_0 * __cse_0d
    sd = __hcse_0 + __hcse_0
    s = __cse_0 * __cse_0
    w = i_w0
    for i_p = 1:i_npass
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_0] = w
        for i_j = 1:w
            __idx_s_stack_1_0 = prefix_s_stack_1[(i_p - 1) + 1] + ((i_j - 1) + 1)
            s_stack_d[__idx_s_stack_1_0] = sd
            s_stack[__idx_s_stack_1_0] = s
            __hcse_1 = u[i_j]
            __hcse_2 = x[i_j]
            sd = __hcse_1 * xd[i_j] + __hcse_2 * ud[i_j]
            s = __hcse_2 * __hcse_1
        end
        __hcse_3 = s * sd
        outd[i_p] = outd[i_p] + (__hcse_3 + __hcse_3)
        out[i_p] = out[i_p] + s * s
        w = w - 3
        __idx_s_stack_1_5 = (prefix_s_stack_1[(i_p - 1) + 1] + max(0, div(val_w_1[(i_p - 1) + 1] - 1, 1) + 1)) + 1
        s_stack_d[__idx_s_stack_1_5] = sd
        s_stack[__idx_s_stack_1_5] = s
    end
    __ihcse_4 = __tot_s_stack_1 + 1
    __idx_s_stack_3 = __ihcse_4
    s_stack_d[__idx_s_stack_3] = sd
    s_stack[__idx_s_stack_3] = s
    __idx_s_stack_0 = __ihcse_4
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_p = i_npass:-1:1
        __idx_s_stack_1_0 = (prefix_s_stack_1[(i_p - 1) + 1] + max(0, div(val_w_1[(i_p - 1) + 1] - 1, 1) + 1)) + 1
        sd = s_stack_d[__idx_s_stack_1_0]
        s = s_stack[__idx_s_stack_1_0]
        __hcse_5 = outb[i_p]
        __cse_1d = __hcse_5 * sd + s * outbd[i_p]
        __cse_1 = s * __hcse_5
        sbd = sbd + __cse_1d
        sb = sb + __cse_1
        sbd = sbd + __cse_1d
        sb = sb + __cse_1
        __idx_tripcount_stack_1_4 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        w = tripcount_stack[__idx_tripcount_stack_1_4]
        for i_j = w:-1:1
            __idx_s_stack_1_0 = prefix_s_stack_1[(i_p - 1) + 1] + ((i_j - 1) + 1)
            sd = s_stack_d[__idx_s_stack_1_0]
            s = s_stack[__idx_s_stack_1_0]
            __oldb_2d = sbd
            __oldb_2 = sb
            sbd = 0.0
            sb = 0.0
            __hcse_6 = u[i_j]
            xbd[i_j] = xbd[i_j] + (__oldb_2 * ud[i_j] + __hcse_6 * __oldb_2d)
            xb[i_j] = xb[i_j] + __hcse_6 * __oldb_2
            __hcse_7 = x[i_j]
            ubd[i_j] = ubd[i_j] + (__oldb_2 * xd[i_j] + __hcse_7 * __oldb_2d)
            ub[i_j] = ub[i_j] + __hcse_7 * __oldb_2
        end
    end
    __oldb_0d = sbd
    __oldb_0 = sb
    sbd = 0.0
    sb = 0.0
    __hcse_8 = x[1]
    __cse_2d = __oldb_0 * xd[1] + __hcse_8 * __oldb_0d
    __cse_2 = __hcse_8 * __oldb_0
    xbd[1] = xbd[1] + __cse_2d
    xb[1] = xb[1] + __cse_2
    xbd[1] = xbd[1] + __cse_2d
    xb[1] = xb[1] + __cse_2
    return nothing
end

function entry_empty(x, u, i_npass, i_w0, out)
    s = x[1] * x[1]
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            s = x[i_j] * u[i_j]
        end
        out[i_p] = out[i_p] + s * s
        w = w - 3
    end
end
