function initstacks_early_stage_offset_b(nu, num_levels)
    __mult_2 = 0
    __mult_1 = 0
    nl = 2
    nc = nl - 1
    __mult_1 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
    __mult_2 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
            end
        end
    end
    prefix_branch_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_branch_stack_1 = 0
    prefix_cl_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_cl_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_tripcount_stack_1 = 0
    prefix_u_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_u_stack_1 = 0
    val_nl_1 = Vector{Int64}(undef, max(0, div(num_levels - 2, 1) + 1))
    for i_level = 2:num_levels
        prefix_branch_stack_1[(i_level - 2) + 1] = __tot_branch_stack_1
        prefix_cl_stack_1[(i_level - 2) + 1] = __tot_cl_stack_1
        prefix_tripcount_stack_1[(i_level - 2) + 1] = __tot_tripcount_stack_1
        prefix_u_stack_1[(i_level - 2) + 1] = __tot_u_stack_1
        nl = nl * 2
        nc = nl - 1
        val_nl_1[(i_level - 2) + 1] = nl
        __tot_branch_stack_1 = __tot_branch_stack_1 + max(0, div(nl - 1, 1) + 1)
        __tot_cl_stack_1 = __tot_cl_stack_1 + max(0, div(nl - 1, 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
        __tot_u_stack_1 = __tot_u_stack_1 + max(0, div(nl - 1, 1) + 1)
    end
    tripcount_stack = Vector{Int64}(undef, max(0, div(nu - 1, 1) + 1) + __tot_tripcount_stack_1)
    branch_stack = Vector{Int64}(undef, __mult_1 + __tot_branch_stack_1)
    u_stack = Vector{Float64}(undef, __tot_u_stack_1)
    cl_stack = Vector{Float64}(undef, __tot_cl_stack_1)
    left_stack = Vector{Float64}(undef, __mult_2)
    return (tripcount_stack, branch_stack, u_stack, cl_stack, left_stack, prefix_branch_stack_1, prefix_cl_stack_1, prefix_tripcount_stack_1, prefix_u_stack_1, __mult_1, __mult_2, __tot_branch_stack_1, __tot_cl_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, val_nl_1)
end

function early_stage_offset_b(u, ub, rhs, rhsb, nu, num_levels, tripcount_stack, branch_stack, u_stack, cl_stack, left_stack, prefix_branch_stack_1, prefix_cl_stack_1, prefix_tripcount_stack_1, prefix_u_stack_1, __mult_1, __mult_2, __tot_branch_stack_1, __tot_cl_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, val_nl_1)
    cl = 0.0
    left = 0.0
    clb = 0.0
    leftb = 0.0
    nl = 2
    nc = nl - 1
    for i_k = 1:nu
        __idx_tripcount_stack_0 = (i_k - 1) + 1
        tripcount_stack[__idx_tripcount_stack_0] = nc
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                __idx_branch_stack_0 = ((i_k - 1) + (i_j - 1) * (div(nu - 1, 1) + 1)) + 1
                branch_stack[__idx_branch_stack_0] = 1
                left = u[i_j - 1, 1]
            else
                __idx_branch_stack_0 = ((i_k - 1) + (i_j - 1) * (div(nu - 1, 1) + 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            u[i_j, 1] = 0.5 * (rhs[i_j, 1] + left) * left
            __idx_left_stack_3 = ((i_k - 1) + (i_j - 1) * (div(nu - 1, 1) + 1)) + 1
            left_stack[__idx_left_stack_3] = left
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        nc = nl - 1
        __idx_tripcount_stack_1_2 = (max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1
        tripcount_stack[__idx_tripcount_stack_1_2] = nl
        for j = 1:nl
            cl = 0.0
            if j > 1
                __idx_branch_stack_1_0 = (__mult_1 + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                cl = u[j - 1, i_level - 1]
            else
                __idx_branch_stack_1_0 = (__mult_1 + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            __icse_0 = (j - 1) + 1
            __idx_u_stack_1_2 = prefix_u_stack_1[(i_level - 2) + 1] + __icse_0
            u_stack[__idx_u_stack_1_2] = u[j, i_level]
            u[j, i_level] = u[j, i_level - 1] * cl
            __idx_cl_stack_1_5 = prefix_cl_stack_1[(i_level - 2) + 1] + __icse_0
            cl_stack[__idx_cl_stack_1_5] = cl
        end
    end
    for i_level = num_levels:-1:2
        __idx_tripcount_stack_1_0 = (max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1
        nl = tripcount_stack[__idx_tripcount_stack_1_0]
        for j = nl:-1:1
            __icse_1 = (j - 1) + 1
            __idx_cl_stack_1_0 = prefix_cl_stack_1[(i_level - 2) + 1] + __icse_1
            cl = cl_stack[__idx_cl_stack_1_0]
            __idx_u_stack_1_0 = prefix_u_stack_1[(i_level - 2) + 1] + __icse_1
            u[j, i_level] = u_stack[__idx_u_stack_1_0]
            __oldb_2 = ub[j, i_level]
            ub[j, i_level] = 0.0
            ub[j, i_level - 1] = ub[j, i_level - 1] + cl * __oldb_2
            clb = clb + u[j, i_level - 1] * __oldb_2
            __idx_branch_stack_1_8 = (__mult_1 + prefix_branch_stack_1[(i_level - 2) + 1]) + __icse_1
            __branch = branch_stack[__idx_branch_stack_1_8]
            if __branch == 1
                __oldb_0 = clb
                clb = 0.0
                ub[j - 1, i_level - 1] = ub[j - 1, i_level - 1] + __oldb_0
            end
            clb = 0.0
        end
    end
    for i_k = nu:-1:1
        __idx_tripcount_stack_0 = (i_k - 1) + 1
        nc = tripcount_stack[__idx_tripcount_stack_0]
        for i_j = nc:-1:1
            __icse_2 = ((i_k - 1) + (i_j - 1) * (div(nu - 1, 1) + 1)) + 1
            __idx_left_stack_0 = __icse_2
            left = left_stack[__idx_left_stack_0]
            __oldb_0 = ub[i_j, 1]
            ub[i_j, 1] = 0.0
            __cse_3 = (0.5left) * __oldb_0
            rhsb[i_j, 1] = rhsb[i_j, 1] + __cse_3
            leftb = leftb + __cse_3
            leftb = leftb + (0.5 * (rhs[i_j, 1] + left)) * __oldb_0
            __idx_branch_stack_7 = __icse_2
            __branch = branch_stack[__idx_branch_stack_7]
            if __branch == 1
                __oldb_0 = leftb
                leftb = 0.0
                ub[i_j - 1, 1] = ub[i_j - 1, 1] + __oldb_0
            end
            leftb = 0.0
        end
    end
    return nothing
end

function early_stage_offset(u, rhs, nu, num_levels)
    nl = 2
    nc = nl - 1
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                left = u[i_j - 1, 1]
            end
            u[i_j, 1] = 0.5 * (rhs[i_j, 1] + left) * left
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        nc = nl - 1
        for j = 1:nl
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level - 1]
            end
            u[j, i_level] = u[j, i_level - 1] * cl
        end
    end
end
