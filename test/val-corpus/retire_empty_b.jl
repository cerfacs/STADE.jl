function initstacks_retire_empty_b(i_npass, i_w0)
    w = i_w0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(i_npass - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    prefix_u_stack_1 = Vector{Int}(undef, max(0, div(i_npass - 1, 1) + 1))
    __tot_u_stack_1 = 0
    val_w_1 = Vector{Int64}(undef, max(0, div(i_npass - 1, 1) + 1))
    for i_p = 1:i_npass
        prefix_tripcount_stack_1[(i_p - 1) + 1] = __tot_tripcount_stack_1
        prefix_u_stack_1[(i_p - 1) + 1] = __tot_u_stack_1
        val_w_1[(i_p - 1) + 1] = w
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
        __tot_u_stack_1 = __tot_u_stack_1 + max(0, div(w - 1, 1) + 1)
        w = w - 3
    end
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    u_stack = Vector{Float64}(undef, __tot_u_stack_1)
    return (tripcount_stack, u_stack, prefix_tripcount_stack_1, prefix_u_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, val_w_1)
end

function retire_empty_b(u, ub, x, xb, i_npass, i_w0, tripcount_stack, u_stack, prefix_tripcount_stack_1, prefix_u_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, val_w_1)
    w = i_w0
    for i_p = 1:i_npass
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_0] = w
        for i_j = 1:w
            __idx_u_stack_1_0 = prefix_u_stack_1[(i_p - 1) + 1] + ((i_j - 1) + 1)
            __cse_0 = u[i_j]
            u_stack[__idx_u_stack_1_0] = __cse_0
            u[i_j] = __cse_0 + x[i_j] * __cse_0
        end
        w = w - 3
    end
    for i_p = i_npass:-1:1
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        w = tripcount_stack[__idx_tripcount_stack_1_0]
        for i_j = w:-1:1
            __idx_u_stack_1_0 = prefix_u_stack_1[(i_p - 1) + 1] + ((i_j - 1) + 1)
            u[i_j] = u_stack[__idx_u_stack_1_0]
            __oldb_2 = ub[i_j]
            ub[i_j] = 0.0
            ub[i_j] = ub[i_j] + __oldb_2
            xb[i_j] = xb[i_j] + u[i_j] * __oldb_2
            ub[i_j] = ub[i_j] + x[i_j] * __oldb_2
        end
    end
    return nothing
end

function retire_empty(u, x, i_npass, i_w0)
    w = i_w0
    for i_p = 1:i_npass
        for i_j = 1:w
            u[i_j] = u[i_j] + x[i_j] * u[i_j]
        end
        w = w - 3
    end
end
