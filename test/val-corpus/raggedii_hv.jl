function initstacks_raggedii_b(n)
    prefix_s_stack_1 = Vector{Int}(undef, max(0, div(n - 1, 1) + 1))
    __tot_s_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(n - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    for i_x = 1:n
        prefix_s_stack_1[(i_x - 1) + 1] = __tot_s_stack_1
        prefix_tripcount_stack_1[(i_x - 1) + 1] = __tot_tripcount_stack_1
        s = 0.0
        __tot_s_stack_1 = __tot_s_stack_1 + (max(0, div(2 - 1, 1) + 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + max(0, div(2 - 1, 1) + 1)
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    s_stack = Vector{Float64}(undef, __tot_s_stack_1)
    return (tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
end

function raggedii_hv(out, outb, u, ub, x, xb, n, m0, outd, outbd, ud, ubd, xd, xbd, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_x = 1:n
        sd = 0.0
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_1] = w
            for i_j = 1:w
                __cse_1 = u[i_x]
                __cse_2 = x[i_j]
                sd = sd + (__cse_1 * xd[i_j] + __cse_2 * ud[i_x])
                s = s + __cse_2 * __cse_1
            end
            __idx_s_stack_1_4 = prefix_s_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            s_stack_d[__idx_s_stack_1_4] = sd
            s_stack[__idx_s_stack_1_4] = s
        end
        __cse_3 = s * sd
        outd[i_x] = __cse_3 + __cse_3
        out[i_x] = s * s
        __idx_s_stack_1_3 = (prefix_s_stack_1[(i_x - 1) + 1] + max(0, div(2 - 1, 1) + 1)) + 1
        s_stack_d[__idx_s_stack_1_3] = sd
        s_stack[__idx_s_stack_1_3] = s
    end
    for i_x = n:-1:1
        __idx_s_stack_1_0 = (prefix_s_stack_1[(i_x - 1) + 1] + max(0, div(2 - 1, 1) + 1)) + 1
        sd = s_stack_d[__idx_s_stack_1_0]
        s = s_stack[__idx_s_stack_1_0]
        __cse_4 = outb[i_x]
        __cse_0d = __cse_4 * sd + s * outbd[i_x]
        __cse_0 = s * __cse_4
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        outbd[i_x] = 0.0
        outb[i_x] = 0.0
        for i_y = 2:-1:1
            __idx_s_stack_1_0 = prefix_s_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            sd = s_stack_d[__idx_s_stack_1_0]
            s = s_stack[__idx_s_stack_1_0]
            w = m0 + i_y
            __idx_tripcount_stack_1_3 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            w = tripcount_stack[__idx_tripcount_stack_1_3]
            for i_j = w:-1:1
                __cse_5 = u[i_x]
                xbd[i_j] = xbd[i_j] + (sb * ud[i_x] + __cse_5 * sbd)
                xb[i_j] = xb[i_j] + __cse_5 * sb
                __cse_6 = x[i_j]
                ubd[i_x] = ubd[i_x] + (sb * xd[i_j] + __cse_6 * sbd)
                ub[i_x] = ub[i_x] + __cse_6 * sb
            end
        end
        sbd = 0.0
        sb = 0.0
    end
    return nothing
end

function raggedii(out, u, x, n, m0)
    for i_x = 1:n
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            for i_j = 1:w
                s = s + x[i_j] * u[i_x]
            end
        end
        out[i_x] = s * s
    end
end
