function initstacks_hoist_sentinel_b(i_n)
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(i_n - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    for i_r = 1:i_n
        prefix_tripcount_stack_1[(i_r - 1) + 1] = __tot_tripcount_stack_1
        i_b = mod(i_r, 3)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    return (tripcount_stack, prefix_tripcount_stack_1, __tot_tripcount_stack_1)
end

function hoist_sentinel_hv(y, yb, x, xb, w, wb, i_n, yd, ybd, xd, xbd, wd, wbd, tripcount_stack, prefix_tripcount_stack_1, __tot_tripcount_stack_1)
    for i_r = 1:i_n
        i_b = mod(i_r, 3)
        __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_r - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_1] = i_b
        for i_e = 1:i_b
            __hcse_0 = w[i_e]
            __hcse_1 = x[(i_r - 1) * 2 + i_e]
            yd[i_b] = yd[i_b] + (__hcse_0 * xd[(i_r - 1) * 2 + i_e] + __hcse_1 * wd[i_e])
            y[i_b] = y[i_b] + __hcse_1 * __hcse_0
        end
    end
    for i_r = i_n:-1:1
        i_b = mod(i_r, 3)
        __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_r - 1) + 1] + 1
        i_b = tripcount_stack[__idx_tripcount_stack_1_1]
        for i_e = i_b:-1:1
            __cse_0d = ybd[i_b]
            __cse_0 = yb[i_b]
            __hcse_2 = w[i_e]
            xbd[(i_r - 1) * 2 + i_e] = xbd[(i_r - 1) * 2 + i_e] + (__cse_0 * wd[i_e] + __hcse_2 * __cse_0d)
            xb[(i_r - 1) * 2 + i_e] = xb[(i_r - 1) * 2 + i_e] + __hcse_2 * __cse_0
            __hcse_3 = x[(i_r - 1) * 2 + i_e]
            wbd[i_e] = wbd[i_e] + (__cse_0 * xd[(i_r - 1) * 2 + i_e] + __hcse_3 * __cse_0d)
            wb[i_e] = wb[i_e] + __hcse_3 * __cse_0
        end
    end
    return nothing
end

function hoist_sentinel(y, x, w, i_n)
    for i_r = 1:i_n
        i_b = mod(i_r, 3)
        for i_e = 1:i_b
            y[i_b] = y[i_b] + x[(i_r - 1) * 2 + i_e] * w[i_e]
        end
    end
end
