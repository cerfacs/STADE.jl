function initstacks_raggedind_b(m0, n)
    prefix_s_stack_1 = Vector{Int}(undef, max(0, div(n - 1, 1) + 1))
    __tot_s_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(n - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    for i_x = 1:n
        prefix_s_stack_1[(i_x - 1) + 1] = __tot_s_stack_1
        prefix_tripcount_stack_1[(i_x - 1) + 1] = __tot_tripcount_stack_1
        w = m0 + i_x
        s = 0.0
        __tot_s_stack_1 = __tot_s_stack_1 + 1
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    s_stack = Vector{Float64}(undef, __tot_s_stack_1)
    return (tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
end

function raggedind_hv(out, outb, u, ub, x, xb, n, m0, outd, outbd, ud, ubd, xd, xbd, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_x = 1:n
        w = m0 + i_x
        sd = 0.0
        s = 0.0
        __idx_tripcount_stack_1_2 = prefix_tripcount_stack_1[(i_x - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_2] = w
        for i_j = 1:w
            __hcse_0 = u[i_x]
            __hcse_1 = x[i_j]
            sd = sd + (__hcse_0 * xd[i_j] + __hcse_1 * ud[i_x])
            s = s + __hcse_1 * __hcse_0
        end
        __hcse_2 = s * sd
        outd[i_x] = __hcse_2 + __hcse_2
        out[i_x] = s * s
        __idx_s_stack_1_6 = prefix_s_stack_1[(i_x - 1) + 1] + 1
        s_stack_d[__idx_s_stack_1_6] = sd
        s_stack[__idx_s_stack_1_6] = s
    end
    for i_x = n:-1:1
        __idx_s_stack_1_0 = prefix_s_stack_1[(i_x - 1) + 1] + 1
        sd = s_stack_d[__idx_s_stack_1_0]
        s = s_stack[__idx_s_stack_1_0]
        w = m0 + i_x
        __oldb_0d = outbd[i_x]
        __oldb_0 = outb[i_x]
        outbd[i_x] = 0.0
        outb[i_x] = 0.0
        __cse_0d = __oldb_0 * sd + s * __oldb_0d
        __cse_0 = s * __oldb_0
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        __idx_tripcount_stack_1_7 = prefix_tripcount_stack_1[(i_x - 1) + 1] + 1
        w = tripcount_stack[__idx_tripcount_stack_1_7]
        for i_j = w:-1:1
            __hcse_3 = u[i_x]
            xbd[i_j] = xbd[i_j] + (sb * ud[i_x] + __hcse_3 * sbd)
            xb[i_j] = xb[i_j] + __hcse_3 * sb
            __hcse_4 = x[i_j]
            ubd[i_x] = ubd[i_x] + (sb * xd[i_j] + __hcse_4 * sbd)
            ub[i_x] = ub[i_x] + __hcse_4 * sb
        end
        sbd = 0.0
        sb = 0.0
    end
    return nothing
end

function raggedind(out, u, x, n, m0)
    for i_x = 1:n
        w = m0 + i_x
        s = 0.0
        for i_j = 1:w
            s = s + x[i_j] * u[i_x]
        end
        out[i_x] = s * s
    end
end
