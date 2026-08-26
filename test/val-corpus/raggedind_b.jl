function initstacks_raggedind_b(m0, n)
    prefix_s_stack_1 = Vector{Int}(undef, div(n - 1, 1) + 1)
    __tot_s_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, div(n - 1, 1) + 1)
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
    s_stack = Vector{Float64}(undef, __tot_s_stack_1 + 1)
    return (tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
end

function raggedind_b(out, outb, u, ub, x, xb, n, m0, tripcount_stack, s_stack, prefix_s_stack_1, prefix_tripcount_stack_1, __tot_s_stack_1, __tot_tripcount_stack_1)
    s = 0.0
    sb = 0.0
    for i_x = 1:n
        w = m0 + i_x
        s = 0.0
        __idx_tripcount_stack_1_2 = prefix_tripcount_stack_1[(i_x - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_2] = w
        for i_j = 1:w
            s = s + x[i_j] * u[i_x]
        end
        out[i_x] = s * s
        __idx_s_stack_1_6 = prefix_s_stack_1[(i_x - 1) + 1] + 1
        s_stack[__idx_s_stack_1_6] = s
    end
    s_stack[__tot_s_stack_1 + 1] = s
    s = s_stack[__tot_s_stack_1 + 1]
    for i_x = n:-1:1
        __idx_s_stack_1_0 = prefix_s_stack_1[(i_x - 1) + 1] + 1
        s = s_stack[__idx_s_stack_1_0]
        w = m0 + i_x
        sb = sb + s * outb[i_x]
        sb = sb + s * outb[i_x]
        outb[i_x] = 0.0
        __idx_tripcount_stack_1_6 = prefix_tripcount_stack_1[(i_x - 1) + 1] + 1
        w = tripcount_stack[__idx_tripcount_stack_1_6]
        for i_j = w:-1:1
            xb[i_j] = xb[i_j] + u[i_x] * sb
            ub[i_x] = ub[i_x] + x[i_j] * sb
        end
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
