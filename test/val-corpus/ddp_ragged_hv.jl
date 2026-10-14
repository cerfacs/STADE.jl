function initstacks_ddp_ragged_b(i_n)
    prefix_s_stack_1 = Vector{Int}(undef, max(0, div(i_n - 1, 1) + 1))
    __tot_s_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(i_n - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    for i_x = 1:i_n
        prefix_s_stack_1[(i_x - 1) + 1] = __tot_s_stack_1
        prefix_tripcount_stack_1[(i_x - 1) + 1] = __tot_tripcount_stack_1
        s = 0.0
        __tot_s_stack_1 = __tot_s_stack_1 + (max(0, div(2 - 1, 1) + 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + max(0, div(2 - 1, 1) + 1)
    end
    for i_x2 = 1:i_n
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    s_stack = Vector{Float64}(undef, __tot_s_stack_1)
    return (tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
end

function ddp_ragged_hv(loss, lossb, u, ub, a, ab, v, vb, i_n, m0, lossd, lossbd, ud, ubd, ad, abd, vd, vbd, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    for i_x = 1:i_n
        sd = 0.0
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_1] = w
            for i_j = 1:w
                __hcse_0 = u[i_x]
                __hcse_1 = a[i_j]
                sd = sd + (__hcse_0 * ad[i_j] + __hcse_1 * ud[i_x])
                s = s + __hcse_1 * __hcse_0
            end
            __idx_s_stack_1_4 = prefix_s_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            s_stack_d[__idx_s_stack_1_4] = sd
            s_stack[__idx_s_stack_1_4] = s
        end
        __hcse_2 = s * sd
        vd[i_x] = __hcse_2 + __hcse_2
        v[i_x] = s * s
        __idx_s_stack_1_3 = (prefix_s_stack_1[(i_x - 1) + 1] + max(0, div(2 - 1, 1) + 1)) + 1
        s_stack_d[__idx_s_stack_1_3] = sd
        s_stack[__idx_s_stack_1_3] = s
    end
    for i_x2 = 1:i_n
        lossd[1] = lossd[1] + vd[i_x2]
        loss[1] = loss[1] + v[i_x2]
    end
    for i_x2 = i_n:-1:1
        vbd[i_x2] = vbd[i_x2] + lossbd[1]
        vb[i_x2] = vb[i_x2] + lossb[1]
    end
    for i_x = i_n:-1:1
        __idx_s_stack_1_0 = (prefix_s_stack_1[(i_x - 1) + 1] + max(0, div(2 - 1, 1) + 1)) + 1
        sd = s_stack_d[__idx_s_stack_1_0]
        s = s_stack[__idx_s_stack_1_0]
        __oldb_0d = vbd[i_x]
        __oldb_0 = vb[i_x]
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
        __cse_0d = __oldb_0 * sd + s * __oldb_0d
        __cse_0 = s * __oldb_0
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        sbd = sbd + __cse_0d
        sb = sb + __cse_0
        for i_y = 2:-1:1
            __icse_1 = (i_y - 1) + 1
            __idx_s_stack_1_0 = prefix_s_stack_1[(i_x - 1) + 1] + __icse_1
            sd = s_stack_d[__idx_s_stack_1_0]
            s = s_stack[__idx_s_stack_1_0]
            w = m0 + i_y
            __idx_tripcount_stack_1_3 = prefix_tripcount_stack_1[(i_x - 1) + 1] + __icse_1
            w = tripcount_stack[__idx_tripcount_stack_1_3]
            for i_j = w:-1:1
                __hcse_3 = u[i_x]
                abd[i_j] = abd[i_j] + (sb * ud[i_x] + __hcse_3 * sbd)
                ab[i_j] = ab[i_j] + __hcse_3 * sb
                __hcse_4 = a[i_j]
                ubd[i_x] = ubd[i_x] + (sb * ad[i_j] + __hcse_4 * sbd)
                ub[i_x] = ub[i_x] + __hcse_4 * sb
            end
        end
        sbd = 0.0
        sb = 0.0
    end
    return nothing
end

function ddp_ragged(loss, u, a, v, i_n, m0)
    for i_x = 1:i_n
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            for i_j = 1:w
                s = s + a[i_j] * u[i_x]
            end
        end
        v[i_x] = s * s
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2]
    end
end
