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

function cascadic_mg_prolong_hv(u, ub, rhs, rhsb, h_coarse, h_coarseb, nu, num_levels, ud, ubd, rhsd, rhsbd, h_coarsed, h_coarsebd, tripcount_stack, branch_stack, hl_stack, hl2_stack, cl_stack, cr_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_cl_stack_1, prefix_cr_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_cl_stack_1, __tot_cr_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_nc_1, val_ncoarse_1)
    hl_stack_d = Vector{Float64}(undef, length(hl_stack))
    hl2_stack_d = Vector{Float64}(undef, length(hl2_stack))
    cl_stack_d = Vector{Float64}(undef, length(cl_stack))
    cr_stack_d = Vector{Float64}(undef, length(cr_stack))
    left_stack_d = Vector{Float64}(undef, length(left_stack))
    right_stack_d = Vector{Float64}(undef, length(right_stack))
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
    cld = 0.0
    clbd = 0.0
    crd = 0.0
    crbd = 0.0
    hld = 0.0
    hlbd = 0.0
    hl2d = 0.0
    hl2bd = 0.0
    leftd = 0.0
    leftbd = 0.0
    rightd = 0.0
    rightbd = 0.0
    nl = 2
    hld = h_coarsed
    hl = h_coarse
    nc = nl - 1
    __hcse_0 = hl * hld
    hl2d = __hcse_0 + __hcse_0
    hl2 = hl * hl
    for i_k = 1:nu
        __idx_tripcount_stack_0 = (i_k - 1) + 1
        tripcount_stack[__idx_tripcount_stack_0] = nc
        for i_j = 1:nc
            leftd = 0.0
            left = 0.0
            if i_j > 1
                __idx_branch_stack_0 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 1
                leftd = ud[i_j - 1, 1]
                left = u[i_j - 1, 1]
            else
                __idx_branch_stack_0 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            rightd = 0.0
            right = 0.0
            if i_j < nc
                __icse_0 = div(nc - 1, 1) + 1
                __idx_branch_stack_0 = max(0, div(nu - 1, 1) + 1) * max(0, __icse_0) + (((i_k - 1) * __icse_0 + (i_j - 1)) + 1)
                branch_stack[__idx_branch_stack_0] = 1
                rightd = ud[i_j + 1, 1]
                right = u[i_j + 1, 1]
            else
                __icse_1 = div(nc - 1, 1) + 1
                __idx_branch_stack_0 = max(0, div(nu - 1, 1) + 1) * max(0, __icse_1) + (((i_k - 1) * __icse_1 + (i_j - 1)) + 1)
                branch_stack[__idx_branch_stack_0] = 0
            end
            __hcse_1 = rhs[i_j, 1]
            ud[i_j, 1] = 0.5 * (((__hcse_1 * hl2d + hl2 * rhsd[i_j, 1]) + leftd) + rightd)
            u[i_j, 1] = 0.5 * (hl2 * __hcse_1 + left + right)
            __icse_2 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
            __idx_left_stack_5 = __icse_2
            left_stack_d[__idx_left_stack_5] = leftd
            left_stack[__idx_left_stack_5] = left
            __idx_right_stack_7 = __icse_2
            right_stack_d[__idx_right_stack_7] = rightd
            right_stack[__idx_right_stack_7] = right
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        __idx_hl_stack_1_1 = prefix_hl_stack_1[(i_level - 2) + 1] + 1
        hl_stack_d[__idx_hl_stack_1_1] = hld
        hl_stack[__idx_hl_stack_1_1] = hl
        hld = 0.5hld
        hl = hl / 2.0
        nc = nl - 1
        __idx_hl2_stack_1_5 = prefix_hl2_stack_1[(i_level - 2) + 1] + 1
        hl2_stack_d[__idx_hl2_stack_1_5] = hl2d
        hl2_stack[__idx_hl2_stack_1_5] = hl2
        __hcse_2 = hl * hld
        hl2d = __hcse_2 + __hcse_2
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        __idx_tripcount_stack_1_9 = (max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1
        tripcount_stack[__idx_tripcount_stack_1_9] = ncoarse
        for j = 1:ncoarse
            jf = j * 2
            ud[jf, i_level] = ud[j, i_level - 1]
            u[jf, i_level] = u[j, i_level - 1]
        end
        __idx_tripcount_stack_1_12 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1) + 1
        tripcount_stack[__idx_tripcount_stack_1_12] = ncoarse
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cld = 0.0
            cl = 0.0
            if j > 1
                __icse_3 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __idx_branch_stack_1_0 = ((__icse_3 + __icse_3) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                cld = ud[j - 1, i_level - 1]
                cl = u[j - 1, i_level - 1]
            else
                __icse_4 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __idx_branch_stack_1_0 = ((__icse_4 + __icse_4) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            crd = 0.0
            cr = 0.0
            if j <= ncoarse
                __icse_5 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __idx_branch_stack_1_0 = (((__icse_5 + __icse_5) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                crd = ud[j, i_level - 1]
                cr = u[j, i_level - 1]
            else
                __icse_6 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __idx_branch_stack_1_0 = (((__icse_6 + __icse_6) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            ud[jf, i_level] = 0.5 * (cld + crd)
            u[jf, i_level] = 0.5 * (cl + cr)
            __icse_7 = (j - 1) + 1
            __idx_cl_stack_1_6 = prefix_cl_stack_1[(i_level - 2) + 1] + __icse_7
            cl_stack_d[__idx_cl_stack_1_6] = cld
            cl_stack[__idx_cl_stack_1_6] = cl
            __idx_cr_stack_1_8 = prefix_cr_stack_1[(i_level - 2) + 1] + __icse_7
            cr_stack_d[__idx_cr_stack_1_8] = crd
            cr_stack[__idx_cr_stack_1_8] = cr
        end
        for i_k = 1:nu
            __idx_tripcount_stack_1_0 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + (1 + 1)) + ((i_k - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_0] = nc
            for i_j = 1:nc
                leftd = 0.0
                left = 0.0
                if i_j > 1
                    __icse_8 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                    __icse_9 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_1_0 = (((__icse_8 + __icse_8) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_9 + __icse_9)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    leftd = ud[i_j - 1, i_level]
                    left = u[i_j - 1, i_level]
                else
                    __icse_10 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                    __icse_11 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_1_0 = (((__icse_10 + __icse_10) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_11 + __icse_11)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                rightd = 0.0
                right = 0.0
                if i_j < nc
                    __icse_12 = max(0, div(nu - 1, 1) + 1)
                    __icse_13 = __icse_12 * max(0, div(nc - 1, 1) + 1)
                    __icse_14 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __icse_15 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (((__icse_13 + __icse_13) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_14 + __icse_14) + __icse_12 * max(0, __icse_15))) + (((i_k - 1) * __icse_15 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    rightd = ud[i_j + 1, i_level]
                    right = u[i_j + 1, i_level]
                else
                    __icse_16 = max(0, div(nu - 1, 1) + 1)
                    __icse_17 = __icse_16 * max(0, div(nc - 1, 1) + 1)
                    __icse_18 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __icse_19 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (((__icse_17 + __icse_17) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_18 + __icse_18) + __icse_16 * max(0, __icse_19))) + (((i_k - 1) * __icse_19 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                __hcse_3 = rhs[i_j, i_level]
                ud[i_j, i_level] = 0.5 * (((__hcse_3 * hl2d + hl2 * rhsd[i_j, i_level]) + leftd) + rightd)
                u[i_j, i_level] = 0.5 * (hl2 * __hcse_3 + left + right)
                __icse_20 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __icse_21 = ((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1
                __idx_left_stack_1_5 = (__icse_20 + prefix_left_stack_1[(i_level - 2) + 1]) + __icse_21
                left_stack_d[__idx_left_stack_1_5] = leftd
                left_stack[__idx_left_stack_1_5] = left
                __idx_right_stack_1_7 = (__icse_20 + prefix_right_stack_1[(i_level - 2) + 1]) + __icse_21
                right_stack_d[__idx_right_stack_1_7] = rightd
                right_stack[__idx_right_stack_1_7] = right
            end
        end
    end
    __ihcse_4 = __tot_hl_stack_1 + 1
    __idx_hl_stack_6 = __ihcse_4
    hl_stack_d[__idx_hl_stack_6] = hld
    hl_stack[__idx_hl_stack_6] = hl
    __ihcse_5 = __tot_hl2_stack_1 + 1
    __idx_hl2_stack_8 = __ihcse_5
    hl2_stack_d[__idx_hl2_stack_8] = hl2d
    hl2_stack[__idx_hl2_stack_8] = hl2
    __idx_hl_stack_0 = __ihcse_4
    hld = hl_stack_d[__idx_hl_stack_0]
    hl = hl_stack[__idx_hl_stack_0]
    __idx_hl2_stack_2 = __ihcse_5
    hl2d = hl2_stack_d[__idx_hl2_stack_2]
    hl2 = hl2_stack[__idx_hl2_stack_2]
    for i_level = num_levels:-1:2
        for i_k = nu:-1:1
            __idx_tripcount_stack_1_0 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + (1 + 1)) + ((i_k - 1) + 1)
            nc = tripcount_stack[__idx_tripcount_stack_1_0]
            for i_j = nc:-1:1
                __icse_22 = max(0, div(nu - 1, 1) + 1)
                __icse_23 = __icse_22 * max(0, div(nc - 1, 1) + 1)
                __icse_24 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                __icse_25 = ((i_k - 1) * __icse_24 + (i_j - 1)) + 1
                __idx_left_stack_1_0 = (__icse_23 + prefix_left_stack_1[(i_level - 2) + 1]) + __icse_25
                leftd = left_stack_d[__idx_left_stack_1_0]
                left = left_stack[__idx_left_stack_1_0]
                __idx_right_stack_1_2 = (__icse_23 + prefix_right_stack_1[(i_level - 2) + 1]) + __icse_25
                rightd = right_stack_d[__idx_right_stack_1_2]
                right = right_stack[__idx_right_stack_1_2]
                __icse_26 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                __idx_branch_stack_1_4 = (((__icse_23 + __icse_23) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_26 + __icse_26) + __icse_22 * max(0, __icse_24))) + __icse_25
                __branch_pre_4 = branch_stack[__idx_branch_stack_1_4]
                rightd = 0.0
                right = 0.0
                if __branch_pre_4 == 1
                    rightd = ud[i_j + 1, i_level]
                    right = u[i_j + 1, i_level]
                else
                    rightd = 0.0
                    right = 0.0
                end
                __icse_27 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
                __icse_28 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                __idx_branch_stack_1_8 = (((__icse_27 + __icse_27) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_28 + __icse_28)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                __branch_pre_2 = branch_stack[__idx_branch_stack_1_8]
                leftd = 0.0
                left = 0.0
                if __branch_pre_2 == 1
                    leftd = ud[i_j - 1, i_level]
                    left = u[i_j - 1, i_level]
                else
                    leftd = 0.0
                    left = 0.0
                end
                __oldb_0d = ubd[i_j, i_level]
                __oldb_0 = ub[i_j, i_level]
                ubd[i_j, i_level] = 0.0
                ub[i_j, i_level] = 0.0
                __cse_29d = 0.5__oldb_0d
                __cse_29 = 0.5__oldb_0
                __hcse_6 = rhs[i_j, i_level]
                hl2bd = hl2bd + (__cse_29 * rhsd[i_j, i_level] + __hcse_6 * __cse_29d)
                hl2b = hl2b + __hcse_6 * __cse_29
                rhsbd[i_j, i_level] = rhsbd[i_j, i_level] + (__cse_29 * hl2d + hl2 * __cse_29d)
                rhsb[i_j, i_level] = rhsb[i_j, i_level] + hl2 * __cse_29
                leftbd = leftbd + __cse_29d
                leftb = leftb + __cse_29
                rightbd = rightbd + __cse_29d
                rightb = rightb + __cse_29
                if __branch_pre_4 == 1
                    __oldb_0d = rightbd
                    __oldb_0 = rightb
                    rightbd = 0.0
                    rightb = 0.0
                    ubd[i_j + 1, i_level] = ubd[i_j + 1, i_level] + __oldb_0d
                    ub[i_j + 1, i_level] = ub[i_j + 1, i_level] + __oldb_0
                end
                rightbd = 0.0
                rightb = 0.0
                if __branch_pre_2 == 1
                    __oldb_0d = leftbd
                    __oldb_0 = leftb
                    leftbd = 0.0
                    leftb = 0.0
                    ubd[i_j - 1, i_level] = ubd[i_j - 1, i_level] + __oldb_0d
                    ub[i_j - 1, i_level] = ub[i_j - 1, i_level] + __oldb_0
                end
                leftbd = 0.0
                leftb = 0.0
            end
        end
        __idx_tripcount_stack_1_1 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1) + 1
        ncoarse = tripcount_stack[__idx_tripcount_stack_1_1]
        for j = ncoarse + 1:-1:1
            __icse_30 = (j - 1) + 1
            __idx_cl_stack_1_0 = prefix_cl_stack_1[(i_level - 2) + 1] + __icse_30
            cld = cl_stack_d[__idx_cl_stack_1_0]
            cl = cl_stack[__idx_cl_stack_1_0]
            __idx_cr_stack_1_2 = prefix_cr_stack_1[(i_level - 2) + 1] + __icse_30
            crd = cr_stack_d[__idx_cr_stack_1_2]
            cr = cr_stack[__idx_cr_stack_1_2]
            jf = j * 2 - 1
            __icse_31 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
            __idx_branch_stack_1_5 = (((__icse_31 + __icse_31) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + __icse_30
            __branch_pre_5 = branch_stack[__idx_branch_stack_1_5]
            crd = 0.0
            cr = 0.0
            if __branch_pre_5 == 1
                crd = ud[j, i_level - 1]
                cr = u[j, i_level - 1]
            else
                crd = 0.0
                cr = 0.0
            end
            __icse_32 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
            __idx_branch_stack_1_9 = ((__icse_32 + __icse_32) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
            __branch_pre_3 = branch_stack[__idx_branch_stack_1_9]
            cld = 0.0
            cl = 0.0
            if __branch_pre_3 == 1
                cld = ud[j - 1, i_level - 1]
                cl = u[j - 1, i_level - 1]
            else
                cld = 0.0
                cl = 0.0
            end
            __oldb_0d = ubd[jf, i_level]
            __oldb_0 = ub[jf, i_level]
            ubd[jf, i_level] = 0.0
            ub[jf, i_level] = 0.0
            __cse_33d = 0.5__oldb_0d
            __cse_33 = 0.5__oldb_0
            clbd = clbd + __cse_33d
            clb = clb + __cse_33
            crbd = crbd + __cse_33d
            crb = crb + __cse_33
            if __branch_pre_5 == 1
                __oldb_0d = crbd
                __oldb_0 = crb
                crbd = 0.0
                crb = 0.0
                ubd[j, i_level - 1] = ubd[j, i_level - 1] + __oldb_0d
                ub[j, i_level - 1] = ub[j, i_level - 1] + __oldb_0
            end
            crbd = 0.0
            crb = 0.0
            if __branch_pre_3 == 1
                __oldb_0d = clbd
                __oldb_0 = clb
                clbd = 0.0
                clb = 0.0
                ubd[j - 1, i_level - 1] = ubd[j - 1, i_level - 1] + __oldb_0d
                ub[j - 1, i_level - 1] = ub[j - 1, i_level - 1] + __oldb_0
            end
            clbd = 0.0
            clb = 0.0
        end
        __idx_tripcount_stack_1_4 = (max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1
        ncoarse = tripcount_stack[__idx_tripcount_stack_1_4]
        for j = ncoarse:-1:1
            jf = j * 2
            __oldb_0d = ubd[jf, i_level]
            __oldb_0 = ub[jf, i_level]
            ubd[jf, i_level] = 0.0
            ub[jf, i_level] = 0.0
            ubd[j, i_level - 1] = ubd[j, i_level - 1] + __oldb_0d
            ub[j, i_level - 1] = ub[j, i_level - 1] + __oldb_0
        end
        __idx_hl2_stack_1_0 = prefix_hl2_stack_1[(i_level - 2) + 1] + 1
        hl2d = hl2_stack_d[__idx_hl2_stack_1_0]
        hl2 = hl2_stack[__idx_hl2_stack_1_0]
        __oldb_2d = hl2bd
        __oldb_2 = hl2b
        hl2bd = 0.0
        hl2b = 0.0
        __cse_34d = __oldb_2 * hld + hl * __oldb_2d
        __cse_34 = hl * __oldb_2
        hlbd = hlbd + __cse_34d
        hlb = hlb + __cse_34
        hlbd = hlbd + __cse_34d
        hlb = hlb + __cse_34
        __idx_hl_stack_1_0 = prefix_hl_stack_1[(i_level - 2) + 1] + 1
        hld = hl_stack_d[__idx_hl_stack_1_0]
        hl = hl_stack[__idx_hl_stack_1_0]
        __oldb_2d = hlbd
        __oldb_2 = hlb
        hlbd = 0.0
        hlb = 0.0
        hlbd = hlbd + 0.5__oldb_2d
        hlb = hlb + 0.5__oldb_2
    end
    for i_k = nu:-1:1
        __idx_tripcount_stack_0 = (i_k - 1) + 1
        nc = tripcount_stack[__idx_tripcount_stack_0]
        for i_j = nc:-1:1
            __icse_35 = div(nc - 1, 1) + 1
            __icse_36 = ((i_k - 1) * __icse_35 + (i_j - 1)) + 1
            __idx_left_stack_0 = __icse_36
            leftd = left_stack_d[__idx_left_stack_0]
            left = left_stack[__idx_left_stack_0]
            __idx_right_stack_2 = __icse_36
            rightd = right_stack_d[__idx_right_stack_2]
            right = right_stack[__idx_right_stack_2]
            __idx_branch_stack_4 = max(0, div(nu - 1, 1) + 1) * max(0, __icse_35) + __icse_36
            __branch_pre_4 = branch_stack[__idx_branch_stack_4]
            rightd = 0.0
            right = 0.0
            if __branch_pre_4 == 1
                rightd = ud[i_j + 1, 1]
                right = u[i_j + 1, 1]
            else
                rightd = 0.0
                right = 0.0
            end
            __idx_branch_stack_8 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
            __branch_pre_2 = branch_stack[__idx_branch_stack_8]
            leftd = 0.0
            left = 0.0
            if __branch_pre_2 == 1
                leftd = ud[i_j - 1, 1]
                left = u[i_j - 1, 1]
            else
                leftd = 0.0
                left = 0.0
            end
            __oldb_0d = ubd[i_j, 1]
            __oldb_0 = ub[i_j, 1]
            ubd[i_j, 1] = 0.0
            ub[i_j, 1] = 0.0
            __cse_37d = 0.5__oldb_0d
            __cse_37 = 0.5__oldb_0
            __hcse_7 = rhs[i_j, 1]
            hl2bd = hl2bd + (__cse_37 * rhsd[i_j, 1] + __hcse_7 * __cse_37d)
            hl2b = hl2b + __hcse_7 * __cse_37
            rhsbd[i_j, 1] = rhsbd[i_j, 1] + (__cse_37 * hl2d + hl2 * __cse_37d)
            rhsb[i_j, 1] = rhsb[i_j, 1] + hl2 * __cse_37
            leftbd = leftbd + __cse_37d
            leftb = leftb + __cse_37
            rightbd = rightbd + __cse_37d
            rightb = rightb + __cse_37
            if __branch_pre_4 == 1
                __oldb_0d = rightbd
                __oldb_0 = rightb
                rightbd = 0.0
                rightb = 0.0
                ubd[i_j + 1, 1] = ubd[i_j + 1, 1] + __oldb_0d
                ub[i_j + 1, 1] = ub[i_j + 1, 1] + __oldb_0
            end
            rightbd = 0.0
            rightb = 0.0
            if __branch_pre_2 == 1
                __oldb_0d = leftbd
                __oldb_0 = leftb
                leftbd = 0.0
                leftb = 0.0
                ubd[i_j - 1, 1] = ubd[i_j - 1, 1] + __oldb_0d
                ub[i_j - 1, 1] = ub[i_j - 1, 1] + __oldb_0
            end
            leftbd = 0.0
            leftb = 0.0
        end
    end
    __oldb_0d = hl2bd
    __oldb_0 = hl2b
    hl2bd = 0.0
    hl2b = 0.0
    __cse_38d = __oldb_0 * hld + hl * __oldb_0d
    __cse_38 = hl * __oldb_0
    hlbd = hlbd + __cse_38d
    hlb = hlb + __cse_38
    hlbd = hlbd + __cse_38d
    hlb = hlb + __cse_38
    __oldb_0d = hlbd
    __oldb_0 = hlb
    hlbd = 0.0
    hlb = 0.0
    h_coarsebd = h_coarsebd + __oldb_0d
    h_coarseb = h_coarseb + __oldb_0
    return (h_coarseb, h_coarsebd)
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
