function initstacks_cascadic_mg_prolong_b(h_coarse, nu, num_levels)
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
            end
            right = 0.0
            if i_j < nc
            end
        end
    end
    prefix_branch_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_branch_stack_1 = 0
    prefix_cl_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_cl_stack_1 = 0
    prefix_cr_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_cr_stack_1 = 0
    prefix_hl2_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_hl2_stack_1 = 0
    prefix_hl_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_hl_stack_1 = 0
    prefix_left_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_left_stack_1 = 0
    prefix_right_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_right_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_tripcount_stack_1 = 0
    val_nc_1 = Vector{Int64}(undef, max(0, div(num_levels - 2, 1) + 1))
    val_ncoarse_1 = Vector{Int64}(undef, max(0, div(num_levels - 2, 1) + 1))
    for i_level = 2:num_levels
        prefix_branch_stack_1[(i_level - 2) + 1] = __tot_branch_stack_1
        prefix_cl_stack_1[(i_level - 2) + 1] = __tot_cl_stack_1
        prefix_cr_stack_1[(i_level - 2) + 1] = __tot_cr_stack_1
        prefix_hl2_stack_1[(i_level - 2) + 1] = __tot_hl2_stack_1
        prefix_hl_stack_1[(i_level - 2) + 1] = __tot_hl_stack_1
        prefix_left_stack_1[(i_level - 2) + 1] = __tot_left_stack_1
        prefix_right_stack_1[(i_level - 2) + 1] = __tot_right_stack_1
        prefix_tripcount_stack_1[(i_level - 2) + 1] = __tot_tripcount_stack_1
        nl = nl * 2
        hl = hl / 2.0
        nc = nl - 1
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        val_nc_1[(i_level - 2) + 1] = nc
        val_ncoarse_1[(i_level - 2) + 1] = ncoarse
        __tot_branch_stack_1 = __tot_branch_stack_1 + (((max(0, div((ncoarse + 1) - 1, 1) + 1) + max(0, div((ncoarse + 1) - 1, 1) + 1)) + max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)) + max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1))
        __tot_cl_stack_1 = __tot_cl_stack_1 + max(0, div((ncoarse + 1) - 1, 1) + 1)
        __tot_cr_stack_1 = __tot_cr_stack_1 + max(0, div((ncoarse + 1) - 1, 1) + 1)
        __tot_hl2_stack_1 = __tot_hl2_stack_1 + 1
        __tot_hl_stack_1 = __tot_hl_stack_1 + 1
        __tot_left_stack_1 = __tot_left_stack_1 + max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
        __tot_right_stack_1 = __tot_right_stack_1 + max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + ((1 + 1) + max(0, div(nu - 1, 1) + 1))
    end
    tripcount_stack = Vector{Int64}(undef, max(0, div(nu - 1, 1) + 1) + __tot_tripcount_stack_1)
    branch_stack = Vector{Int64}(undef, (max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1) + max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)) + __tot_branch_stack_1)
    hl_stack = Vector{Float64}(undef, __tot_hl_stack_1 + 1)
    hl2_stack = Vector{Float64}(undef, __tot_hl2_stack_1 + 1)
    cl_stack = Vector{Float64}(undef, __tot_cl_stack_1)
    cr_stack = Vector{Float64}(undef, __tot_cr_stack_1)
    left_stack = Vector{Float64}(undef, max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1) + __tot_left_stack_1)
    right_stack = Vector{Float64}(undef, max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1) + __tot_right_stack_1)
    return (tripcount_stack, branch_stack, hl_stack, hl2_stack, cl_stack, cr_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_cl_stack_1, prefix_cr_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_cl_stack_1, __tot_cr_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_nc_1, val_ncoarse_1)
end

function cascadic_mg_prolong_b(u, ub, rhs, rhsb, h_coarse, h_coarseb, nu, num_levels, tripcount_stack, branch_stack, hl_stack, hl2_stack, cl_stack, cr_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_cl_stack_1, prefix_cr_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_cl_stack_1, __tot_cr_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_nc_1, val_ncoarse_1)
    cl = 0.0
    cr = 0.0
    hl = 0.0
    hl2 = 0.0
    left = 0.0
    right = 0.0
    clb = 0.0
    crb = 0.0
    hlb = 0.0
    hl2b = 0.0
    leftb = 0.0
    rightb = 0.0
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_k = 1:nu
        __idx_tripcount_stack_0 = (i_k - 1) + 1
        tripcount_stack[__idx_tripcount_stack_0] = nc
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                __idx_branch_stack_0 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 1
                left = u[i_j - 1, 1]
            else
                __idx_branch_stack_0 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            right = 0.0
            if i_j < nc
                __icse_0 = div(nc - 1, 1) + 1
                __idx_branch_stack_0 = max(0, div(nu - 1, 1) + 1) * max(0, __icse_0) + (((i_k - 1) * __icse_0 + (i_j - 1)) + 1)
                branch_stack[__idx_branch_stack_0] = 1
                right = u[i_j + 1, 1]
            else
                __icse_1 = div(nc - 1, 1) + 1
                __idx_branch_stack_0 = max(0, div(nu - 1, 1) + 1) * max(0, __icse_1) + (((i_k - 1) * __icse_1 + (i_j - 1)) + 1)
                branch_stack[__idx_branch_stack_0] = 0
            end
            u[i_j, 1] = 0.5 * (hl2 * rhs[i_j, 1] + left + right)
            __icse_2 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
            __idx_left_stack_5 = __icse_2
            left_stack[__idx_left_stack_5] = left
            __idx_right_stack_7 = __icse_2
            right_stack[__idx_right_stack_7] = right
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        __idx_hl_stack_1_1 = prefix_hl_stack_1[(i_level - 2) + 1] + 1
        hl_stack[__idx_hl_stack_1_1] = hl
        hl = hl / 2.0
        nc = nl - 1
        __idx_hl2_stack_1_5 = prefix_hl2_stack_1[(i_level - 2) + 1] + 1
        hl2_stack[__idx_hl2_stack_1_5] = hl2
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        __idx_tripcount_stack_1_9 = (max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1
        tripcount_stack[__idx_tripcount_stack_1_9] = ncoarse
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_level] = u[j, i_level - 1]
        end
        __idx_tripcount_stack_1_12 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1) + 1
        tripcount_stack[__idx_tripcount_stack_1_12] = ncoarse
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                __icse_3 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __idx_branch_stack_1_0 = ((__icse_3 + __icse_3) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                cl = u[j - 1, i_level - 1]
            else
                __icse_4 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __idx_branch_stack_1_0 = ((__icse_4 + __icse_4) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            cr = 0.0
            if j <= ncoarse
                __icse_5 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __idx_branch_stack_1_0 = (((__icse_5 + __icse_5) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                cr = u[j, i_level - 1]
            else
                __icse_6 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __idx_branch_stack_1_0 = (((__icse_6 + __icse_6) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            u[jf, i_level] = 0.5 * (cl + cr)
            __icse_7 = (j - 1) + 1
            __idx_cl_stack_1_6 = prefix_cl_stack_1[(i_level - 2) + 1] + __icse_7
            cl_stack[__idx_cl_stack_1_6] = cl
            __idx_cr_stack_1_8 = prefix_cr_stack_1[(i_level - 2) + 1] + __icse_7
            cr_stack[__idx_cr_stack_1_8] = cr
        end
        for i_k = 1:nu
            __idx_tripcount_stack_1_0 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + (1 + 1)) + ((i_k - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_0] = nc
            for i_j = 1:nc
                left = 0.0
                if i_j > 1
                    __icse_8 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                    __icse_9 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_1_0 = (((__icse_8 + __icse_8) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_9 + __icse_9)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    left = u[i_j - 1, i_level]
                else
                    __icse_10 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                    __icse_11 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_1_0 = (((__icse_10 + __icse_10) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_11 + __icse_11)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                right = 0.0
                if i_j < nc
                    __icse_12 = max(0, div(nu - 1, 1) + 1)
                    __icse_13 = __icse_12 * max(0, div(nc - 1, 1) + 1)
                    __icse_14 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __icse_15 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (((__icse_13 + __icse_13) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_14 + __icse_14) + __icse_12 * max(0, __icse_15))) + (((i_k - 1) * __icse_15 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    right = u[i_j + 1, i_level]
                else
                    __icse_16 = max(0, div(nu - 1, 1) + 1)
                    __icse_17 = __icse_16 * max(0, div(nc - 1, 1) + 1)
                    __icse_18 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __icse_19 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (((__icse_17 + __icse_17) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_18 + __icse_18) + __icse_16 * max(0, __icse_19))) + (((i_k - 1) * __icse_19 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                u[i_j, i_level] = 0.5 * (hl2 * rhs[i_j, i_level] + left + right)
                __icse_20 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __icse_21 = ((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1
                __idx_left_stack_1_5 = (__icse_20 + prefix_left_stack_1[(i_level - 2) + 1]) + __icse_21
                left_stack[__idx_left_stack_1_5] = left
                __idx_right_stack_1_7 = (__icse_20 + prefix_right_stack_1[(i_level - 2) + 1]) + __icse_21
                right_stack[__idx_right_stack_1_7] = right
            end
        end
    end
    __icse_22 = __tot_hl_stack_1 + 1
    __idx_hl_stack_6 = __icse_22
    hl_stack[__idx_hl_stack_6] = hl
    __icse_23 = __tot_hl2_stack_1 + 1
    __idx_hl2_stack_8 = __icse_23
    hl2_stack[__idx_hl2_stack_8] = hl2
    __idx_hl_stack_0 = __icse_22
    hl = hl_stack[__idx_hl_stack_0]
    __idx_hl2_stack_2 = __icse_23
    hl2 = hl2_stack[__idx_hl2_stack_2]
    for i_level = num_levels:-1:2
        for i_k = nu:-1:1
            __idx_tripcount_stack_1_0 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + (1 + 1)) + ((i_k - 1) + 1)
            nc = tripcount_stack[__idx_tripcount_stack_1_0]
            for i_j = nc:-1:1
                __icse_24 = max(0, div(nu - 1, 1) + 1)
                __icse_25 = __icse_24 * max(0, div(nc - 1, 1) + 1)
                __icse_26 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                __icse_27 = ((i_k - 1) * __icse_26 + (i_j - 1)) + 1
                __idx_left_stack_1_0 = (__icse_25 + prefix_left_stack_1[(i_level - 2) + 1]) + __icse_27
                left = left_stack[__idx_left_stack_1_0]
                __idx_right_stack_1_2 = (__icse_25 + prefix_right_stack_1[(i_level - 2) + 1]) + __icse_27
                right = right_stack[__idx_right_stack_1_2]
                __icse_28 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                __idx_branch_stack_1_4 = (((__icse_25 + __icse_25) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_28 + __icse_28) + __icse_24 * max(0, __icse_26))) + __icse_27
                __branch_pre_4 = branch_stack[__idx_branch_stack_1_4]
                right = 0.0
                if __branch_pre_4 == 1
                    right = u[i_j + 1, i_level]
                else
                    right = 0.0
                end
                __icse_29 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __icse_30 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                __idx_branch_stack_1_8 = (((__icse_29 + __icse_29) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_30 + __icse_30)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                __branch_pre_2 = branch_stack[__idx_branch_stack_1_8]
                left = 0.0
                if __branch_pre_2 == 1
                    left = u[i_j - 1, i_level]
                else
                    left = 0.0
                end
                __oldb_0 = ub[i_j, i_level]
                ub[i_j, i_level] = 0.0
                __cse_31 = 0.5__oldb_0
                hl2b = hl2b + rhs[i_j, i_level] * __cse_31
                rhsb[i_j, i_level] = rhsb[i_j, i_level] + hl2 * __cse_31
                leftb = leftb + __cse_31
                rightb = rightb + __cse_31
                if __branch_pre_4 == 1
                    __oldb_0 = rightb
                    rightb = 0.0
                    ub[i_j + 1, i_level] = ub[i_j + 1, i_level] + __oldb_0
                end
                rightb = 0.0
                if __branch_pre_2 == 1
                    __oldb_0 = leftb
                    leftb = 0.0
                    ub[i_j - 1, i_level] = ub[i_j - 1, i_level] + __oldb_0
                end
                leftb = 0.0
            end
        end
        __idx_tripcount_stack_1_1 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1) + 1
        ncoarse = tripcount_stack[__idx_tripcount_stack_1_1]
        for j = ncoarse + 1:-1:1
            __icse_32 = (j - 1) + 1
            __idx_cl_stack_1_0 = prefix_cl_stack_1[(i_level - 2) + 1] + __icse_32
            cl = cl_stack[__idx_cl_stack_1_0]
            __idx_cr_stack_1_2 = prefix_cr_stack_1[(i_level - 2) + 1] + __icse_32
            cr = cr_stack[__idx_cr_stack_1_2]
            jf = j * 2 - 1
            __icse_33 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
            __idx_branch_stack_1_5 = (((__icse_33 + __icse_33) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + __icse_32
            __branch_pre_5 = branch_stack[__idx_branch_stack_1_5]
            cr = 0.0
            if __branch_pre_5 == 1
                cr = u[j, i_level - 1]
            else
                cr = 0.0
            end
            __icse_34 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
            __idx_branch_stack_1_9 = ((__icse_34 + __icse_34) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
            __branch_pre_3 = branch_stack[__idx_branch_stack_1_9]
            cl = 0.0
            if __branch_pre_3 == 1
                cl = u[j - 1, i_level - 1]
            else
                cl = 0.0
            end
            __oldb_0 = ub[jf, i_level]
            ub[jf, i_level] = 0.0
            __cse_35 = 0.5__oldb_0
            clb = clb + __cse_35
            crb = crb + __cse_35
            if __branch_pre_5 == 1
                __oldb_0 = crb
                crb = 0.0
                ub[j, i_level - 1] = ub[j, i_level - 1] + __oldb_0
            end
            crb = 0.0
            if __branch_pre_3 == 1
                __oldb_0 = clb
                clb = 0.0
                ub[j - 1, i_level - 1] = ub[j - 1, i_level - 1] + __oldb_0
            end
            clb = 0.0
        end
        __idx_tripcount_stack_1_4 = (max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1
        ncoarse = tripcount_stack[__idx_tripcount_stack_1_4]
        for j = ncoarse:-1:1
            jf = j * 2
            __oldb_0 = ub[jf, i_level]
            ub[jf, i_level] = 0.0
            ub[j, i_level - 1] = ub[j, i_level - 1] + __oldb_0
        end
        __idx_hl2_stack_1_0 = prefix_hl2_stack_1[(i_level - 2) + 1] + 1
        hl2 = hl2_stack[__idx_hl2_stack_1_0]
        __oldb_2 = hl2b
        hl2b = 0.0
        __cse_36 = hl * __oldb_2
        hlb = hlb + __cse_36
        hlb = hlb + __cse_36
        __idx_hl_stack_1_0 = prefix_hl_stack_1[(i_level - 2) + 1] + 1
        hl = hl_stack[__idx_hl_stack_1_0]
        __oldb_2 = hlb
        hlb = 0.0
        hlb = hlb + 0.5__oldb_2
    end
    for i_k = nu:-1:1
        __idx_tripcount_stack_0 = (i_k - 1) + 1
        nc = tripcount_stack[__idx_tripcount_stack_0]
        for i_j = nc:-1:1
            __icse_37 = div(nc - 1, 1) + 1
            __icse_38 = ((i_k - 1) * __icse_37 + (i_j - 1)) + 1
            __idx_left_stack_0 = __icse_38
            left = left_stack[__idx_left_stack_0]
            __idx_right_stack_2 = __icse_38
            right = right_stack[__idx_right_stack_2]
            __idx_branch_stack_4 = max(0, div(nu - 1, 1) + 1) * max(0, __icse_37) + __icse_38
            __branch_pre_4 = branch_stack[__idx_branch_stack_4]
            right = 0.0
            if __branch_pre_4 == 1
                right = u[i_j + 1, 1]
            else
                right = 0.0
            end
            __idx_branch_stack_8 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
            __branch_pre_2 = branch_stack[__idx_branch_stack_8]
            left = 0.0
            if __branch_pre_2 == 1
                left = u[i_j - 1, 1]
            else
                left = 0.0
            end
            __oldb_0 = ub[i_j, 1]
            ub[i_j, 1] = 0.0
            __cse_39 = 0.5__oldb_0
            hl2b = hl2b + rhs[i_j, 1] * __cse_39
            rhsb[i_j, 1] = rhsb[i_j, 1] + hl2 * __cse_39
            leftb = leftb + __cse_39
            rightb = rightb + __cse_39
            if __branch_pre_4 == 1
                __oldb_0 = rightb
                rightb = 0.0
                ub[i_j + 1, 1] = ub[i_j + 1, 1] + __oldb_0
            end
            rightb = 0.0
            if __branch_pre_2 == 1
                __oldb_0 = leftb
                leftb = 0.0
                ub[i_j - 1, 1] = ub[i_j - 1, 1] + __oldb_0
            end
            leftb = 0.0
        end
    end
    __oldb_0 = hl2b
    hl2b = 0.0
    __cse_40 = hl * __oldb_0
    hlb = hlb + __cse_40
    hlb = hlb + __cse_40
    __oldb_0 = hlb
    hlb = 0.0
    h_coarseb = h_coarseb + __oldb_0
    return h_coarseb
end

function cascadic_mg_prolong(u, rhs, h_coarse, nu, num_levels)
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                left = u[i_j - 1, 1]
            end
            right = 0.0
            if i_j < nc
                right = u[i_j + 1, 1]
            end
            u[i_j, 1] = 0.5 * (hl2 * rhs[i_j, 1] + left + right)
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        hl = hl / 2.0
        nc = nl - 1
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_level] = u[j, i_level - 1]
        end
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level - 1]
            end
            cr = 0.0
            if j <= ncoarse
                cr = u[j, i_level - 1]
            end
            u[jf, i_level] = 0.5 * (cl + cr)
        end
        for i_k = 1:nu
            for i_j = 1:nc
                left = 0.0
                if i_j > 1
                    left = u[i_j - 1, i_level]
                end
                right = 0.0
                if i_j < nc
                    right = u[i_j + 1, i_level]
                end
                u[i_j, i_level] = 0.5 * (hl2 * rhs[i_j, i_level] + left + right)
            end
        end
    end
end
