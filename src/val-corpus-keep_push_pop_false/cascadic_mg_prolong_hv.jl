function initstacks_cascadic_mg_prolong_b(h_coarse, nu, num_levels)
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_seq_k = 1:nu
        for i_seq_j = 1:nc
            left = 0.0
            if i_seq_j > 1
            end
            right = 0.0
            if i_seq_j < nc
            end
        end
    end
    prefix_branch_stack_1 = Vector{Int}(undef, div(num_levels - 2, 1) + 1)
    __tot_branch_stack_1 = 0
    prefix_cl_stack_1 = Vector{Int}(undef, div(num_levels - 2, 1) + 1)
    __tot_cl_stack_1 = 0
    prefix_cr_stack_1 = Vector{Int}(undef, div(num_levels - 2, 1) + 1)
    __tot_cr_stack_1 = 0
    prefix_hl2_stack_1 = Vector{Int}(undef, div(num_levels - 2, 1) + 1)
    __tot_hl2_stack_1 = 0
    prefix_hl_stack_1 = Vector{Int}(undef, div(num_levels - 2, 1) + 1)
    __tot_hl_stack_1 = 0
    prefix_left_stack_1 = Vector{Int}(undef, div(num_levels - 2, 1) + 1)
    __tot_left_stack_1 = 0
    prefix_right_stack_1 = Vector{Int}(undef, div(num_levels - 2, 1) + 1)
    __tot_right_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, div(num_levels - 2, 1) + 1)
    __tot_tripcount_stack_1 = 0
    val_nc_1 = Vector{Int64}(undef, div(num_levels - 2, 1) + 1)
    val_ncoarse_1 = Vector{Int64}(undef, div(num_levels - 2, 1) + 1)
    for i_seq_level = 2:num_levels
        prefix_branch_stack_1[(i_seq_level - 2) + 1] = __tot_branch_stack_1
        prefix_cl_stack_1[(i_seq_level - 2) + 1] = __tot_cl_stack_1
        prefix_cr_stack_1[(i_seq_level - 2) + 1] = __tot_cr_stack_1
        prefix_hl2_stack_1[(i_seq_level - 2) + 1] = __tot_hl2_stack_1
        prefix_hl_stack_1[(i_seq_level - 2) + 1] = __tot_hl_stack_1
        prefix_left_stack_1[(i_seq_level - 2) + 1] = __tot_left_stack_1
        prefix_right_stack_1[(i_seq_level - 2) + 1] = __tot_right_stack_1
        prefix_tripcount_stack_1[(i_seq_level - 2) + 1] = __tot_tripcount_stack_1
        nl = nl * 2
        hl = hl / 2.0
        nc = nl - 1
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        val_nc_1[(i_seq_level - 2) + 1] = nc
        val_ncoarse_1[(i_seq_level - 2) + 1] = ncoarse
        __tot_branch_stack_1 = __tot_branch_stack_1 + ((((div((ncoarse + 1) - 1, 1) + 1) + (div((ncoarse + 1) - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1))
        __tot_cl_stack_1 = __tot_cl_stack_1 + ((((div((ncoarse + 1) - 1, 1) + 1) + (div((ncoarse + 1) - 1, 1) + 1)) + (div((ncoarse + 1) - 1, 1) + 1)) + 1)
        __tot_cr_stack_1 = __tot_cr_stack_1 + ((((div((ncoarse + 1) - 1, 1) + 1) + (div((ncoarse + 1) - 1, 1) + 1)) + (div((ncoarse + 1) - 1, 1) + 1)) + 1)
        __tot_hl2_stack_1 = __tot_hl2_stack_1 + 1
        __tot_hl_stack_1 = __tot_hl_stack_1 + 1
        __tot_left_stack_1 = __tot_left_stack_1 + (((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + 1)
        __tot_right_stack_1 = __tot_right_stack_1 + (((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + ((1 + 1) + (div(nu - 1, 1) + 1))
    end
    hl_stack = Vector{Float64}(undef, (1 + __tot_hl_stack_1) + 1)
    hl2_stack = Vector{Float64}(undef, (1 + __tot_hl2_stack_1) + 1)
    tripcount_stack = Vector{Int64}(undef, (div(nu - 1, 1) + 1) + __tot_tripcount_stack_1)
    left_stack = Vector{Float64}(undef, (((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_left_stack_1) + 1)
    branch_stack = Vector{Int64}(undef, ((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + __tot_branch_stack_1)
    right_stack = Vector{Float64}(undef, (((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_right_stack_1) + 1)
    cl_stack = Vector{Float64}(undef, __tot_cl_stack_1 + 1)
    cr_stack = Vector{Float64}(undef, __tot_cr_stack_1 + 1)
    return (hl_stack, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, cl_stack, cr_stack, prefix_branch_stack_1, prefix_cl_stack_1, prefix_cr_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_cl_stack_1, __tot_cr_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_nc_1, val_ncoarse_1)
end

function cascadic_mg_prolong_hv(u, ub, rhs, rhsb, h_coarse, h_coarseb, nu, num_levels, ud, ubd, rhsd, rhsbd, h_coarsed, h_coarsebd, hl_stack, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, cl_stack, cr_stack, prefix_branch_stack_1, prefix_cl_stack_1, prefix_cr_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_cl_stack_1, __tot_cr_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_nc_1, val_ncoarse_1)
    hl_stack_d = Vector{Float64}(undef, length(hl_stack))
    hl2_stack_d = Vector{Float64}(undef, length(hl2_stack))
    left_stack_d = Vector{Float64}(undef, length(left_stack))
    right_stack_d = Vector{Float64}(undef, length(right_stack))
    cl_stack_d = Vector{Float64}(undef, length(cl_stack))
    cr_stack_d = Vector{Float64}(undef, length(cr_stack))
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
    hl_stack_d[1] = hld
    hl_stack[1] = hl
    hld = h_coarsed
    hl = h_coarse
    nc = nl - 1
    hl2_stack_d[1] = hl2d
    hl2_stack[1] = hl2
    hl2d = hl * hld + hl * hld
    hl2 = hl * hl
    for i_seq_k = 1:nu
        tripcount_stack[(i_seq_k - 1) + 1] = nc
        for i_seq_j = 1:nc
            left_stack_d[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1] = leftd
            left_stack[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1] = left
            leftd = 0.0
            left = 0.0
            if i_seq_j > 1
                branch_stack[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1] = 1
                left_stack_d[(div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = leftd
                left_stack[(div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = left
                leftd = ud[i_seq_j - 1, 1]
                left = u[i_seq_j - 1, 1]
            else
                branch_stack[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1] = 0
            end
            right_stack_d[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1] = rightd
            right_stack[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1] = right
            rightd = 0.0
            right = 0.0
            if i_seq_j < nc
                branch_stack[(div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = 1
                right_stack_d[(div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = rightd
                right_stack[(div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = right
                rightd = ud[i_seq_j + 1, 1]
                right = u[i_seq_j + 1, 1]
            else
                branch_stack[(div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = 0
            end
            ud[i_seq_j, 1] = 0.5 * (((rhs[i_seq_j, 1] * hl2d + hl2 * rhsd[i_seq_j, 1]) + leftd) + rightd)
            u[i_seq_j, 1] = 0.5 * (hl2 * rhs[i_seq_j, 1] + left + right)
            left_stack_d[((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = leftd
            left_stack[((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = left
            right_stack_d[((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = rightd
            right_stack[((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = right
        end
        left_stack_d[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + ((i_seq_k - 1) + 1)] = leftd
        left_stack[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + ((i_seq_k - 1) + 1)] = left
        right_stack_d[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + ((i_seq_k - 1) + 1)] = rightd
        right_stack[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + ((i_seq_k - 1) + 1)] = right
    end
    for i_seq_level = 2:num_levels
        nl = nl * 2
        hl_stack_d[(1 + prefix_hl_stack_1[(i_seq_level - 2) + 1]) + 1] = hld
        hl_stack[(1 + prefix_hl_stack_1[(i_seq_level - 2) + 1]) + 1] = hl
        hld = 0.5hld
        hl = hl / 2.0
        nc = nl - 1
        hl2_stack_d[(1 + prefix_hl2_stack_1[(i_seq_level - 2) + 1]) + 1] = hl2d
        hl2_stack[(1 + prefix_hl2_stack_1[(i_seq_level - 2) + 1]) + 1] = hl2
        hl2d = hl * hld + hl * hld
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        tripcount_stack[((div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_seq_level - 2) + 1]) + 1] = ncoarse
        for j = 1:ncoarse
            jf = j * 2
            ud[jf, i_seq_level] = ud[j, i_seq_level - 1]
            u[jf, i_seq_level] = u[j, i_seq_level - 1]
        end
        tripcount_stack[(((div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_seq_level - 2) + 1]) + 1) + 1] = ncoarse
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cl_stack_d[prefix_cl_stack_1[(i_seq_level - 2) + 1] + ((j - 1) + 1)] = cld
            cl_stack[prefix_cl_stack_1[(i_seq_level - 2) + 1] + ((j - 1) + 1)] = cl
            cld = 0.0
            cl = 0.0
            if j > 1
                branch_stack[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + ((j - 1) + 1)] = 1
                cl_stack_d[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)] = cld
                cl_stack[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)] = cl
                cld = ud[j - 1, i_seq_level - 1]
                cl = u[j - 1, i_seq_level - 1]
            else
                branch_stack[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + ((j - 1) + 1)] = 0
            end
            cr_stack_d[prefix_cr_stack_1[(i_seq_level - 2) + 1] + ((j - 1) + 1)] = crd
            cr_stack[prefix_cr_stack_1[(i_seq_level - 2) + 1] + ((j - 1) + 1)] = cr
            crd = 0.0
            cr = 0.0
            if j <= ncoarse
                branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)] = 1
                cr_stack_d[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)] = crd
                cr_stack[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)] = cr
                crd = ud[j, i_seq_level - 1]
                cr = u[j, i_seq_level - 1]
            else
                branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)] = 0
            end
            ud[jf, i_seq_level] = 0.5 * (cld + crd)
            u[jf, i_seq_level] = 0.5 * (cl + cr)
            cl_stack_d[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)] = cld
            cl_stack[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)] = cl
            cr_stack_d[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)] = crd
            cr_stack[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)] = cr
        end
        for i_seq_k = 1:nu
            tripcount_stack[(((div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_seq_level - 2) + 1]) + (1 + 1)) + ((i_seq_k - 1) + 1)] = nc
            for i_seq_j = 1:nc
                left_stack_d[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = leftd
                left_stack[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = left
                leftd = 0.0
                left = 0.0
                if i_seq_j > 1
                    branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = 1
                    left_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = leftd
                    left_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = left
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = 0
                end
                right_stack_d[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = rightd
                right_stack[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = right
                rightd = 0.0
                right = 0.0
                if i_seq_j < nc
                    branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = 1
                    right_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = rightd
                    right_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = right
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = 0
                end
                ud[i_seq_j, i_seq_level] = 0.5 * (((rhs[i_seq_j, i_seq_level] * hl2d + hl2 * rhsd[i_seq_j, i_seq_level]) + leftd) + rightd)
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * rhs[i_seq_j, i_seq_level] + left + right)
                left_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + ((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = leftd
                left_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + ((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = left
                right_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + ((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = rightd
                right_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + ((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)] = right
            end
            left_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)] = leftd
            left_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)] = left
            right_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)] = rightd
            right_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)] = right
        end
        cl_stack_d[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + 1] = cld
        cl_stack[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + 1] = cl
        cr_stack_d[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + 1] = crd
        cr_stack[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + 1] = cr
        left_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + ((((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1))) + 1] = leftd
        left_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + ((((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1))) + 1] = left
        right_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + ((((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1))) + 1] = rightd
        right_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + ((((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1))) + 1] = right
    end
    cl_stack_d[__tot_cl_stack_1 + 1] = cld
    cl_stack[__tot_cl_stack_1 + 1] = cl
    cr_stack_d[__tot_cr_stack_1 + 1] = crd
    cr_stack[__tot_cr_stack_1 + 1] = cr
    hl_stack_d[(1 + __tot_hl_stack_1) + 1] = hld
    hl_stack[(1 + __tot_hl_stack_1) + 1] = hl
    hl2_stack_d[(1 + __tot_hl2_stack_1) + 1] = hl2d
    hl2_stack[(1 + __tot_hl2_stack_1) + 1] = hl2
    left_stack_d[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_left_stack_1) + 1] = leftd
    left_stack[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_left_stack_1) + 1] = left
    right_stack_d[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_right_stack_1) + 1] = rightd
    right_stack[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_right_stack_1) + 1] = right
    cld = cl_stack_d[__tot_cl_stack_1 + 1]
    cl = cl_stack[__tot_cl_stack_1 + 1]
    crd = cr_stack_d[__tot_cr_stack_1 + 1]
    cr = cr_stack[__tot_cr_stack_1 + 1]
    hld = hl_stack_d[(1 + __tot_hl_stack_1) + 1]
    hl = hl_stack[(1 + __tot_hl_stack_1) + 1]
    hl2d = hl2_stack_d[(1 + __tot_hl2_stack_1) + 1]
    hl2 = hl2_stack[(1 + __tot_hl2_stack_1) + 1]
    leftd = left_stack_d[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_left_stack_1) + 1]
    left = left_stack[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_left_stack_1) + 1]
    rightd = right_stack_d[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_right_stack_1) + 1]
    right = right_stack[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + __tot_right_stack_1) + 1]
    for i_seq_level = num_levels:-1:2
        cld = cl_stack_d[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + 1]
        cl = cl_stack[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + 1]
        crd = cr_stack_d[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + 1]
        cr = cr_stack[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + 1]
        leftd = left_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + ((((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1))) + 1]
        left = left_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + ((((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1))) + 1]
        rightd = right_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + ((((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1))) + 1]
        right = right_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + ((((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1))) + 1]
        for i_seq_k = nu:-1:1
            leftd = left_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)]
            left = left_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)]
            rightd = right_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)]
            right = right_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)]
            nc = tripcount_stack[(((div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_seq_level - 2) + 1]) + (1 + 1)) + ((i_seq_k - 1) + 1)]
            for i_seq_j = nc:-1:1
                leftd = left_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + ((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                left = left_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + ((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                rightd = right_stack_d[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + ((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                right = right_stack[((((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + ((div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                __branch_pre_4 = branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + (((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                rightd = 0.0
                right = 0.0
                if __branch_pre_4 == 1
                    rightd = ud[i_seq_j + 1, i_seq_level]
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    rightd = 0.0
                    right = 0.0
                end
                __branch_pre_2 = branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                leftd = 0.0
                left = 0.0
                if __branch_pre_2 == 1
                    leftd = ud[i_seq_j - 1, i_seq_level]
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    leftd = 0.0
                    left = 0.0
                end
                hl2bd = hl2bd + ((0.5 * ub[i_seq_j, i_seq_level]) * rhsd[i_seq_j, i_seq_level] + rhs[i_seq_j, i_seq_level] * (0.5 * ubd[i_seq_j, i_seq_level]))
                hl2b = hl2b + rhs[i_seq_j, i_seq_level] * (0.5 * ub[i_seq_j, i_seq_level])
                rhsbd[i_seq_j, i_seq_level] = rhsbd[i_seq_j, i_seq_level] + ((0.5 * ub[i_seq_j, i_seq_level]) * hl2d + hl2 * (0.5 * ubd[i_seq_j, i_seq_level]))
                rhsb[i_seq_j, i_seq_level] = rhsb[i_seq_j, i_seq_level] + hl2 * (0.5 * ub[i_seq_j, i_seq_level])
                leftbd = leftbd + 0.5 * ubd[i_seq_j, i_seq_level]
                leftb = leftb + 0.5 * ub[i_seq_j, i_seq_level]
                rightbd = rightbd + 0.5 * ubd[i_seq_j, i_seq_level]
                rightb = rightb + 0.5 * ub[i_seq_j, i_seq_level]
                ubd[i_seq_j, i_seq_level] = 0.0
                ub[i_seq_j, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ubd[i_seq_j + 1, i_seq_level] = ubd[i_seq_j + 1, i_seq_level] + rightbd
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                    rightbd = 0.0
                    rightb = 0.0
                end
                rightd = right_stack_d[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                right = right_stack[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_right_stack_1[(i_seq_level - 2) + 1]) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                rightbd = 0.0
                rightb = 0.0
                if __branch_pre_2 == 1
                    ubd[i_seq_j - 1, i_seq_level] = ubd[i_seq_j - 1, i_seq_level] + leftbd
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftbd = 0.0
                    leftb = 0.0
                end
                leftd = left_stack_d[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                left = left_stack[(((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1)) + prefix_left_stack_1[(i_seq_level - 2) + 1]) + (((i_seq_k - 1) * (div(val_nc_1[(i_seq_level - 2) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
                leftbd = 0.0
                leftb = 0.0
            end
        end
        ncoarse = tripcount_stack[(((div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_seq_level - 2) + 1]) + 1) + 1]
        for j = ncoarse + 1:-1:1
            cld = cl_stack_d[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)]
            cl = cl_stack[(prefix_cl_stack_1[(i_seq_level - 2) + 1] + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)]
            crd = cr_stack_d[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)]
            cr = cr_stack[(prefix_cr_stack_1[(i_seq_level - 2) + 1] + ((div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)]
            jf = j * 2 - 1
            __branch_pre_5 = branch_stack[((((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + (div((val_ncoarse_1[(i_seq_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)]
            crd = 0.0
            cr = 0.0
            if __branch_pre_5 == 1
                crd = ud[j, i_seq_level - 1]
                cr = u[j, i_seq_level - 1]
            else
                crd = 0.0
                cr = 0.0
            end
            __branch_pre_3 = branch_stack[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + prefix_branch_stack_1[(i_seq_level - 2) + 1]) + ((j - 1) + 1)]
            cld = 0.0
            cl = 0.0
            if __branch_pre_3 == 1
                cld = ud[j - 1, i_seq_level - 1]
                cl = u[j - 1, i_seq_level - 1]
            else
                cld = 0.0
                cl = 0.0
            end
            clbd = clbd + 0.5 * ubd[jf, i_seq_level]
            clb = clb + 0.5 * ub[jf, i_seq_level]
            crbd = crbd + 0.5 * ubd[jf, i_seq_level]
            crb = crb + 0.5 * ub[jf, i_seq_level]
            ubd[jf, i_seq_level] = 0.0
            ub[jf, i_seq_level] = 0.0
            if __branch_pre_5 == 1
                ubd[j, i_seq_level - 1] = ubd[j, i_seq_level - 1] + crbd
                ub[j, i_seq_level - 1] = ub[j, i_seq_level - 1] + crb
                crbd = 0.0
                crb = 0.0
            end
            crd = cr_stack_d[prefix_cr_stack_1[(i_seq_level - 2) + 1] + ((j - 1) + 1)]
            cr = cr_stack[prefix_cr_stack_1[(i_seq_level - 2) + 1] + ((j - 1) + 1)]
            crbd = 0.0
            crb = 0.0
            if __branch_pre_3 == 1
                ubd[j - 1, i_seq_level - 1] = ubd[j - 1, i_seq_level - 1] + clbd
                ub[j - 1, i_seq_level - 1] = ub[j - 1, i_seq_level - 1] + clb
                clbd = 0.0
                clb = 0.0
            end
            cld = cl_stack_d[prefix_cl_stack_1[(i_seq_level - 2) + 1] + ((j - 1) + 1)]
            cl = cl_stack[prefix_cl_stack_1[(i_seq_level - 2) + 1] + ((j - 1) + 1)]
            clbd = 0.0
            clb = 0.0
        end
        ncoarse = tripcount_stack[((div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_seq_level - 2) + 1]) + 1]
        for j = 1:ncoarse
            jf = j * 2
            ubd[j, i_seq_level - 1] = ubd[j, i_seq_level - 1] + ubd[jf, i_seq_level]
            ub[j, i_seq_level - 1] = ub[j, i_seq_level - 1] + ub[jf, i_seq_level]
            ubd[jf, i_seq_level] = 0.0
            ub[jf, i_seq_level] = 0.0
        end
        hl2d = hl2_stack_d[(1 + prefix_hl2_stack_1[(i_seq_level - 2) + 1]) + 1]
        hl2 = hl2_stack[(1 + prefix_hl2_stack_1[(i_seq_level - 2) + 1]) + 1]
        hlbd = hlbd + (hl2b * hld + hl * hl2bd)
        hlb = hlb + hl * hl2b
        hlbd = hlbd + (hl2b * hld + hl * hl2bd)
        hlb = hlb + hl * hl2b
        hl2bd = 0.0
        hl2b = 0.0
        hld = hl_stack_d[(1 + prefix_hl_stack_1[(i_seq_level - 2) + 1]) + 1]
        hl = hl_stack[(1 + prefix_hl_stack_1[(i_seq_level - 2) + 1]) + 1]
        hlbd = 0.5hlbd
        hlb = 0.5hlb
    end
    for i_seq_k = nu:-1:1
        leftd = left_stack_d[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + ((i_seq_k - 1) + 1)]
        left = left_stack[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + ((i_seq_k - 1) + 1)]
        rightd = right_stack_d[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + ((i_seq_k - 1) + 1)]
        right = right_stack[(((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + ((i_seq_k - 1) + 1)]
        nc = tripcount_stack[(i_seq_k - 1) + 1]
        for i_seq_j = nc:-1:1
            leftd = left_stack_d[((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
            left = left_stack[((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
            rightd = right_stack_d[((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
            right = right_stack[((div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1)) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
            __branch_pre_4 = branch_stack[(div(nu - 1, 1) + 1) * (div(nc - 1, 1) + 1) + (((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1)]
            rightd = 0.0
            right = 0.0
            if __branch_pre_4 == 1
                rightd = ud[i_seq_j + 1, 1]
                right = u[i_seq_j + 1, 1]
            else
                rightd = 0.0
                right = 0.0
            end
            __branch_pre_2 = branch_stack[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1]
            leftd = 0.0
            left = 0.0
            if __branch_pre_2 == 1
                leftd = ud[i_seq_j - 1, 1]
                left = u[i_seq_j - 1, 1]
            else
                leftd = 0.0
                left = 0.0
            end
            hl2bd = hl2bd + ((0.5 * ub[i_seq_j, 1]) * rhsd[i_seq_j, 1] + rhs[i_seq_j, 1] * (0.5 * ubd[i_seq_j, 1]))
            hl2b = hl2b + rhs[i_seq_j, 1] * (0.5 * ub[i_seq_j, 1])
            rhsbd[i_seq_j, 1] = rhsbd[i_seq_j, 1] + ((0.5 * ub[i_seq_j, 1]) * hl2d + hl2 * (0.5 * ubd[i_seq_j, 1]))
            rhsb[i_seq_j, 1] = rhsb[i_seq_j, 1] + hl2 * (0.5 * ub[i_seq_j, 1])
            leftbd = leftbd + 0.5 * ubd[i_seq_j, 1]
            leftb = leftb + 0.5 * ub[i_seq_j, 1]
            rightbd = rightbd + 0.5 * ubd[i_seq_j, 1]
            rightb = rightb + 0.5 * ub[i_seq_j, 1]
            ubd[i_seq_j, 1] = 0.0
            ub[i_seq_j, 1] = 0.0
            if __branch_pre_4 == 1
                ubd[i_seq_j + 1, 1] = ubd[i_seq_j + 1, 1] + rightbd
                ub[i_seq_j + 1, 1] = ub[i_seq_j + 1, 1] + rightb
                rightbd = 0.0
                rightb = 0.0
            end
            rightd = right_stack_d[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1]
            right = right_stack[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1]
            rightbd = 0.0
            rightb = 0.0
            if __branch_pre_2 == 1
                ubd[i_seq_j - 1, 1] = ubd[i_seq_j - 1, 1] + leftbd
                ub[i_seq_j - 1, 1] = ub[i_seq_j - 1, 1] + leftb
                leftbd = 0.0
                leftb = 0.0
            end
            leftd = left_stack_d[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1]
            left = left_stack[((i_seq_k - 1) * (div(nc - 1, 1) + 1) + (i_seq_j - 1)) + 1]
            leftbd = 0.0
            leftb = 0.0
        end
    end
    hl2d = hl2_stack_d[1]
    hl2 = hl2_stack[1]
    hlbd = hlbd + (hl2b * hld + hl * hl2bd)
    hlb = hlb + hl * hl2b
    hlbd = hlbd + (hl2b * hld + hl * hl2bd)
    hlb = hlb + hl * hl2b
    hl2bd = 0.0
    hl2b = 0.0
    hld = hl_stack_d[1]
    hl = hl_stack[1]
    h_coarsebd = h_coarsebd + hlbd
    h_coarseb = h_coarseb + hlb
    hlbd = 0.0
    hlb = 0.0
    return (h_coarseb, h_coarsebd)
end

function cascadic_mg_prolong(u, rhs, h_coarse, nu, num_levels)
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_seq_k = 1:nu
        for i_seq_j = 1:nc
            left = 0.0
            if i_seq_j > 1
                left = u[i_seq_j - 1, 1]
            end
            right = 0.0
            if i_seq_j < nc
                right = u[i_seq_j + 1, 1]
            end
            u[i_seq_j, 1] = 0.5 * (hl2 * rhs[i_seq_j, 1] + left + right)
        end
    end
    for i_seq_level = 2:num_levels
        nl = nl * 2
        hl = hl / 2.0
        nc = nl - 1
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_seq_level] = u[j, i_seq_level - 1]
        end
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_seq_level - 1]
            end
            cr = 0.0
            if j <= ncoarse
                cr = u[j, i_seq_level - 1]
            end
            u[jf, i_seq_level] = 0.5 * (cl + cr)
        end
        for i_seq_k = 1:nu
            for i_seq_j = 1:nc
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < nc
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * rhs[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
end
