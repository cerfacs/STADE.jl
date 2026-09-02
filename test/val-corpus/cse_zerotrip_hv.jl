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

function cse_zerotrip_hv(out, outb, u, ub, w, wb, i_npass, i_w0, outd, outbd, ud, ubd, wd, wbd, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    __hcse_0 = w[1]
    __hcse_1 = u[1]
    sd = __hcse_0 * ud[1] + __hcse_1 * wd[1]
    s = __hcse_1 * __hcse_0
    width = i_w0
    for i_p = 1:i_npass
        __cse_0d = ud[1]
        __cse_0 = u[1]
        __hcse_2 = w[1]
        sd = sd + (((__hcse_2 * __cse_0) * __cse_0d + (__cse_0 * __cse_0) * wd[1]) + (__cse_0 * __hcse_2) * __cse_0d)
        s = s + __cse_0 * __hcse_2 * __cse_0
        __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_1] = width
        for i_j = 1:width
            __hcse_3 = w[i_j]
            __hcse_4 = u[i_j]
            sd = sd + (__hcse_3 * ud[i_j] + __hcse_4 * wd[i_j])
            s = s + __hcse_4 * __hcse_3
        end
        __hcse_5 = u[1]
        __hcse_6 = w[1]
        __hcse_7 = __hcse_5 * __hcse_6
        outd[i_p] = __hcse_7 * sd + s * (__hcse_6 * ud[1] + __hcse_5 * wd[1])
        out[i_p] = s * __hcse_7
        width = width - 3
        __idx_s_stack_1_6 = prefix_s_stack_1[(i_p - 1) + 1] + 1
        s_stack_d[__idx_s_stack_1_6] = sd
        s_stack[__idx_s_stack_1_6] = s
    end
    __ihcse_8 = __tot_s_stack_1 + 1
    __idx_s_stack_3 = __ihcse_8
    s_stack_d[__idx_s_stack_3] = sd
    s_stack[__idx_s_stack_3] = s
    __idx_s_stack_0 = __ihcse_8
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i_p = i_npass:-1:1
        __idx_s_stack_1_0 = prefix_s_stack_1[(i_p - 1) + 1] + 1
        sd = s_stack_d[__idx_s_stack_1_0]
        s = s_stack[__idx_s_stack_1_0]
        __oldb_0d = outbd[i_p]
        __oldb_0 = outb[i_p]
        outbd[i_p] = 0.0
        outb[i_p] = 0.0
        __cse_1d = ud[1]
        __cse_1 = u[1]
        __cse_2d = wd[1]
        __cse_2 = w[1]
        __hcse_9 = __cse_1 * __cse_2
        sbd = sbd + (__oldb_0 * (__cse_2 * __cse_1d + __cse_1 * __cse_2d) + __hcse_9 * __oldb_0d)
        sb = sb + __hcse_9 * __oldb_0
        __cse_3d = __oldb_0 * sd + s * __oldb_0d
        __cse_3 = s * __oldb_0
        ubd[1] = ubd[1] + (__cse_3 * __cse_2d + __cse_2 * __cse_3d)
        ub[1] = ub[1] + __cse_2 * __cse_3
        wbd[1] = wbd[1] + (__cse_3 * __cse_1d + __cse_1 * __cse_3d)
        wb[1] = wb[1] + __cse_1 * __cse_3
        __idx_tripcount_stack_1_7 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        width = tripcount_stack[__idx_tripcount_stack_1_7]
        for i_j = width:-1:1
            __hcse_10 = w[i_j]
            ubd[i_j] = ubd[i_j] + (sb * wd[i_j] + __hcse_10 * sbd)
            ub[i_j] = ub[i_j] + __hcse_10 * sb
            __hcse_11 = u[i_j]
            wbd[i_j] = wbd[i_j] + (sb * ud[i_j] + __hcse_11 * sbd)
            wb[i_j] = wb[i_j] + __hcse_11 * sb
        end
        __cse_4d = wd[1]
        __cse_4 = w[1]
        __cse_5d = ud[1]
        __cse_5 = u[1]
        __hcse_12 = __cse_5 * __cse_4d
        __hcse_13 = __cse_4 * __cse_5d
        __hcse_14 = __cse_4 * __cse_5
        ubd[1] = ubd[1] + (sb * (__hcse_12 + __hcse_13) + __hcse_14 * sbd)
        ub[1] = ub[1] + __hcse_14 * sb
        __hcse_15 = __cse_5 * __cse_5d
        __hcse_16 = __cse_5 * __cse_5
        wbd[1] = wbd[1] + (sb * (__hcse_15 + __hcse_15) + __hcse_16 * sbd)
        wb[1] = wb[1] + __hcse_16 * sb
        __hcse_17 = __cse_5 * __cse_4
        ubd[1] = ubd[1] + (sb * (__hcse_13 + __hcse_12) + __hcse_17 * sbd)
        ub[1] = ub[1] + __hcse_17 * sb
    end
    __oldb_0d = sbd
    __oldb_0 = sb
    sbd = 0.0
    sb = 0.0
    __hcse_18 = w[1]
    ubd[1] = ubd[1] + (__oldb_0 * wd[1] + __hcse_18 * __oldb_0d)
    ub[1] = ub[1] + __hcse_18 * __oldb_0
    __hcse_19 = u[1]
    wbd[1] = wbd[1] + (__oldb_0 * ud[1] + __hcse_19 * __oldb_0d)
    wb[1] = wb[1] + __hcse_19 * __oldb_0
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
