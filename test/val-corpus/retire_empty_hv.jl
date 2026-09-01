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

function retire_empty_hv(u, ub, x, xb, i_npass, i_w0, ud, ubd, xd, xbd, tripcount_stack, u_stack, prefix_tripcount_stack_1, prefix_u_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, val_w_1)
    u_stack_d = Vector{Float64}(undef, length(u_stack))
    w = i_w0
    for i_p = 1:i_npass
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_0] = w
        for i_j = 1:w
            __idx_u_stack_1_0 = prefix_u_stack_1[(i_p - 1) + 1] + ((i_j - 1) + 1)
            __cse_0d = ud[i_j]
            __cse_0 = u[i_j]
            u_stack_d[__idx_u_stack_1_0] = __cse_0d
            u_stack[__idx_u_stack_1_0] = __cse_0
            __cse_2 = x[i_j]
            ud[i_j] = __cse_0d + (__cse_0 * xd[i_j] + __cse_2 * __cse_0d)
            u[i_j] = __cse_0 + __cse_2 * __cse_0
        end
        w = w - 3
    end
    for i_p = i_npass:-1:1
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_p - 1) + 1] + 1
        w = tripcount_stack[__idx_tripcount_stack_1_0]
        for i_j = w:-1:1
            __idx_u_stack_1_0 = prefix_u_stack_1[(i_p - 1) + 1] + ((i_j - 1) + 1)
            ud[i_j] = u_stack_d[__idx_u_stack_1_0]
            u[i_j] = u_stack[__idx_u_stack_1_0]
            __cse_1d = ubd[i_j]
            __cse_1 = ub[i_j]
            __cse_3 = u[i_j]
            xbd[i_j] = xbd[i_j] + (__cse_1 * ud[i_j] + __cse_3 * __cse_1d)
            xb[i_j] = xb[i_j] + __cse_3 * __cse_1
            __cse_4 = x[i_j]
            ubd[i_j] = __cse_1d + (__cse_1 * xd[i_j] + __cse_4 * __cse_1d)
            ub[i_j] = __cse_1 + __cse_4 * __cse_1
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
