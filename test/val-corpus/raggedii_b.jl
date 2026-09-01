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

function raggedii_b(out, outb, u, ub, x, xb, n, m0, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s = 0.0
    sb = 0.0
    for i_x = 1:n
        s = 0.0
        for i_y = 1:2
            w = m0 + i_y
            __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_1] = w
            for i_j = 1:w
                s = s + x[i_j] * u[i_x]
            end
            __idx_s_stack_1_4 = prefix_s_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            s_stack[__idx_s_stack_1_4] = s
        end
        out[i_x] = s * s
        __idx_s_stack_1_3 = (prefix_s_stack_1[(i_x - 1) + 1] + max(0, div(2 - 1, 1) + 1)) + 1
        s_stack[__idx_s_stack_1_3] = s
    end
    for i_x = n:-1:1
        __idx_s_stack_1_0 = (prefix_s_stack_1[(i_x - 1) + 1] + max(0, div(2 - 1, 1) + 1)) + 1
        s = s_stack[__idx_s_stack_1_0]
        __cse_0 = s * outb[i_x]
        sb = sb + __cse_0
        sb = sb + __cse_0
        outb[i_x] = 0.0
        for i_y = 2:-1:1
            __idx_s_stack_1_0 = prefix_s_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            s = s_stack[__idx_s_stack_1_0]
            w = m0 + i_y
            __idx_tripcount_stack_1_3 = prefix_tripcount_stack_1[(i_x - 1) + 1] + ((i_y - 1) + 1)
            w = tripcount_stack[__idx_tripcount_stack_1_3]
            for i_j = w:-1:1
                xb[i_j] = xb[i_j] + u[i_x] * sb
                ub[i_x] = ub[i_x] + x[i_j] * sb
            end
        end
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
