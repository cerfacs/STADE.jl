function initstacks_mg_vcycle_b(h1, n, nfine, nu1, nu2, num_levels)
    n = n * 2
    nl = nfine
    hl = h1
    prefix_branch_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_branch_stack_1 = 0
    prefix_f_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_f_stack_1 = 0
    prefix_hl2_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_hl2_stack_1 = 0
    prefix_hl_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_hl_stack_1 = 0
    prefix_left_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_left_stack_1 = 0
    prefix_r_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_r_stack_1 = 0
    prefix_right_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_right_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    prefix_u_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_u_stack_1 = 0
    val_n_1 = Vector{Int64}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    val_nc_1 = Vector{Int64}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    for i_level = 1:num_levels - 1
        prefix_branch_stack_1[(i_level - 1) + 1] = __tot_branch_stack_1
        prefix_f_stack_1[(i_level - 1) + 1] = __tot_f_stack_1
        prefix_hl2_stack_1[(i_level - 1) + 1] = __tot_hl2_stack_1
        prefix_hl_stack_1[(i_level - 1) + 1] = __tot_hl_stack_1
        prefix_left_stack_1[(i_level - 1) + 1] = __tot_left_stack_1
        prefix_r_stack_1[(i_level - 1) + 1] = __tot_r_stack_1
        prefix_right_stack_1[(i_level - 1) + 1] = __tot_right_stack_1
        prefix_tripcount_stack_1[(i_level - 1) + 1] = __tot_tripcount_stack_1
        prefix_u_stack_1[(i_level - 1) + 1] = __tot_u_stack_1
        n = nl - 1
        hl2 = hl * hl
        ncg = div(nl, 2)
        nc = ncg - 1
        nl = ncg
        hl = hl * 2.0
        val_n_1[(i_level - 1) + 1] = n
        val_nc_1[(i_level - 1) + 1] = nc
        __tot_branch_stack_1 = __tot_branch_stack_1 + (((max(0, div(nu1 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(nu1 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n - 1, 1) + 1)) + max(0, div(n - 1, 1) + 1))
        __tot_f_stack_1 = __tot_f_stack_1 + max(0, div(nc - 1, 1) + 1)
        __tot_hl2_stack_1 = __tot_hl2_stack_1 + 1
        __tot_hl_stack_1 = __tot_hl_stack_1 + 1
        __tot_left_stack_1 = __tot_left_stack_1 + (max(0, div(nu1 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n - 1, 1) + 1))
        __tot_r_stack_1 = __tot_r_stack_1 + max(0, div(n - 1, 1) + 1)
        __tot_right_stack_1 = __tot_right_stack_1 + (max(0, div(nu1 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(n - 1, 1) + 1))
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + (((max(0, div(nu1 - 1, 1) + 1) + 1) + 1) + 1)
        __tot_u_stack_1 = __tot_u_stack_1 + (max(0, div(nu1 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(nc - 1, 1) + 1))
    end
    hl2 = hl * hl
    prefix_branch_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_branch_stack_2 = 0
    prefix_cl_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_cl_stack_2 = 0
    prefix_cr_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_cr_stack_2 = 0
    prefix_hl2_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_hl2_stack_2 = 0
    prefix_hl_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_hl_stack_2 = 0
    prefix_left_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_left_stack_2 = 0
    prefix_right_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_right_stack_2 = 0
    prefix_tripcount_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_tripcount_stack_2 = 0
    prefix_u_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_u_stack_2 = 0
    val_n_2 = Vector{Int64}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    val_nc_2 = Vector{Int64}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    for i_level = num_levels - 1:-1:1
        prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_branch_stack_2
        prefix_cl_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_cl_stack_2
        prefix_cr_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_cr_stack_2
        prefix_hl2_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_hl2_stack_2
        prefix_hl_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_hl_stack_2
        prefix_left_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_left_stack_2
        prefix_right_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_right_stack_2
        prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_tripcount_stack_2
        prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_u_stack_2
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        val_n_2[div(i_level - (num_levels - 1), -1) + 1] = n
        val_nc_2[div(i_level - (num_levels - 1), -1) + 1] = nc
        __tot_branch_stack_2 = __tot_branch_stack_2 + (((max(0, div((nc + 1) - 1, 1) + 1) + max(0, div((nc + 1) - 1, 1) + 1)) + max(0, div(nu2 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(nu2 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
        __tot_cl_stack_2 = __tot_cl_stack_2 + max(0, div((nc + 1) - 1, 1) + 1)
        __tot_cr_stack_2 = __tot_cr_stack_2 + max(0, div((nc + 1) - 1, 1) + 1)
        __tot_hl2_stack_2 = __tot_hl2_stack_2 + 1
        __tot_hl_stack_2 = __tot_hl_stack_2 + 1
        __tot_left_stack_2 = __tot_left_stack_2 + max(0, div(nu2 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)
        __tot_right_stack_2 = __tot_right_stack_2 + max(0, div(nu2 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)
        __tot_tripcount_stack_2 = __tot_tripcount_stack_2 + ((1 + 1) + max(0, div(nu2 - 1, 1) + 1))
        __tot_u_stack_2 = __tot_u_stack_2 + ((max(0, div(nc - 1, 1) + 1) + max(0, div((nc + 1) - 1, 1) + 1)) + max(0, div(nu2 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    end
    hl2_stack = Vector{Float64}(undef, ((__tot_hl2_stack_1 + 1) + __tot_hl2_stack_2) + 1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1 + __tot_tripcount_stack_2)
    branch_stack = Vector{Int64}(undef, __tot_branch_stack_1 + __tot_branch_stack_2)
    u_stack = Vector{Float64}(undef, (__tot_u_stack_1 + 1) + __tot_u_stack_2)
    r_stack = Vector{Float64}(undef, __tot_r_stack_1)
    f_stack = Vector{Float64}(undef, __tot_f_stack_1)
    hl_stack = Vector{Float64}(undef, (__tot_hl_stack_1 + __tot_hl_stack_2) + 1)
    cl_stack = Vector{Float64}(undef, __tot_cl_stack_2)
    cr_stack = Vector{Float64}(undef, __tot_cr_stack_2)
    left_stack = Vector{Float64}(undef, __tot_left_stack_1 + __tot_left_stack_2)
    right_stack = Vector{Float64}(undef, __tot_right_stack_1 + __tot_right_stack_2)
    return (hl2_stack, tripcount_stack, branch_stack, u_stack, r_stack, f_stack, hl_stack, cl_stack, cr_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_f_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_r_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, prefix_u_stack_1, prefix_branch_stack_2, prefix_cl_stack_2, prefix_cr_stack_2, prefix_hl2_stack_2, prefix_hl_stack_2, prefix_left_stack_2, prefix_right_stack_2, prefix_tripcount_stack_2, prefix_u_stack_2, __tot_branch_stack_1, __tot_f_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_r_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, __tot_branch_stack_2, __tot_cl_stack_2, __tot_cr_stack_2, __tot_hl2_stack_2, __tot_hl_stack_2, __tot_left_stack_2, __tot_right_stack_2, __tot_tripcount_stack_2, __tot_u_stack_2, val_n_1, val_nc_1, val_n_2, val_nc_2)
end

function mg_vcycle_hv(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, ud, ubd, fd, fbd, rd, rbd, h1d, h1bd, hl2_stack, tripcount_stack, branch_stack, u_stack, r_stack, f_stack, hl_stack, cl_stack, cr_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_f_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_r_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, prefix_u_stack_1, prefix_branch_stack_2, prefix_cl_stack_2, prefix_cr_stack_2, prefix_hl2_stack_2, prefix_hl_stack_2, prefix_left_stack_2, prefix_right_stack_2, prefix_tripcount_stack_2, prefix_u_stack_2, __tot_branch_stack_1, __tot_f_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_r_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, __tot_branch_stack_2, __tot_cl_stack_2, __tot_cr_stack_2, __tot_hl2_stack_2, __tot_hl_stack_2, __tot_left_stack_2, __tot_right_stack_2, __tot_tripcount_stack_2, __tot_u_stack_2, val_n_1, val_nc_1, val_n_2, val_nc_2)
    hl2_stack_d = Vector{Float64}(undef, length(hl2_stack))
    u_stack_d = Vector{Float64}(undef, length(u_stack))
    r_stack_d = Vector{Float64}(undef, length(r_stack))
    f_stack_d = Vector{Float64}(undef, length(f_stack))
    hl_stack_d = Vector{Float64}(undef, length(hl_stack))
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
    n = n * 2
    nl = nfine
    hld = h1d
    hl = h1
    for i_level = 1:num_levels - 1
        n = nl - 1
        __idx_hl2_stack_1_1 = prefix_hl2_stack_1[(i_level - 1) + 1] + 1
        hl2_stack_d[__idx_hl2_stack_1_1] = hl2d
        hl2_stack[__idx_hl2_stack_1_1] = hl2
        __hcse_0 = hl * hld
        hl2d = __hcse_0 + __hcse_0
        hl2 = hl * hl
        for i_k = 1:nu1
            __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_level - 1) + 1] + ((i_k - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_0] = n
            for i_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_j > 1
                    __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_level - 1) + 1] + (((i_k - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    leftd = ud[i_j - 1, i_level]
                    left = u[i_j - 1, i_level]
                else
                    __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_level - 1) + 1] + (((i_k - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                rightd = 0.0
                right = 0.0
                if i_j < n
                    __icse_0 = div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, __icse_0)) + (((i_k - 1) * __icse_0 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    rightd = ud[i_j + 1, i_level]
                    right = u[i_j + 1, i_level]
                else
                    __icse_1 = div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, __icse_1)) + (((i_k - 1) * __icse_1 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                __icse_2 = ((i_k - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1
                __idx_u_stack_1_4 = prefix_u_stack_1[(i_level - 1) + 1] + __icse_2
                u_stack_d[__idx_u_stack_1_4] = ud[i_j, i_level]
                u_stack[__idx_u_stack_1_4] = u[i_j, i_level]
                __hcse_1 = f[i_j, i_level]
                ud[i_j, i_level] = 0.5 * (((__hcse_1 * hl2d + hl2 * fd[i_j, i_level]) + leftd) + rightd)
                u[i_j, i_level] = 0.5 * (hl2 * __hcse_1 + left + right)
                __idx_left_stack_1_7 = prefix_left_stack_1[(i_level - 1) + 1] + __icse_2
                left_stack_d[__idx_left_stack_1_7] = leftd
                left_stack[__idx_left_stack_1_7] = left
                __idx_right_stack_1_9 = prefix_right_stack_1[(i_level - 1) + 1] + __icse_2
                right_stack_d[__idx_right_stack_1_9] = rightd
                right_stack[__idx_right_stack_1_9] = right
            end
        end
        __idx_tripcount_stack_1_5 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_5] = n
        for j = 1:n
            leftd = 0.0
            left = 0.0
            if j > 1
                __icse_3 = max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + (__icse_3 + __icse_3)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                leftd = ud[j - 1, i_level]
                left = u[j - 1, i_level]
            else
                __icse_4 = max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + (__icse_4 + __icse_4)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            rightd = 0.0
            right = 0.0
            if j < n
                __icse_5 = max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
                __icse_6 = max(0, div(nu1 - 1, 1) + 1) * __icse_5
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + ((__icse_6 + __icse_6) + __icse_5)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                rightd = ud[j + 1, i_level]
                right = u[j + 1, i_level]
            else
                __icse_7 = max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
                __icse_8 = max(0, div(nu1 - 1, 1) + 1) * __icse_7
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + ((__icse_8 + __icse_8) + __icse_7)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            __icse_9 = (j - 1) + 1
            __idx_r_stack_1_4 = prefix_r_stack_1[(i_level - 1) + 1] + __icse_9
            r_stack_d[__idx_r_stack_1_4] = rd[j, i_level]
            r_stack[__idx_r_stack_1_4] = r[j, i_level]
            __hcse_2 = (2.0 * u[j, i_level] - left) - right
            rd[j, i_level] = fd[j, i_level] + -(((1.0 / hl2) * ((2.0 * ud[j, i_level] + -leftd) + -rightd) + -(__hcse_2 / hl2 ^ 2) * hl2d))
            r[j, i_level] = f[j, i_level] - __hcse_2 / hl2
            __icse_10 = max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
            __idx_left_stack_1_7 = (prefix_left_stack_1[(i_level - 1) + 1] + __icse_10) + __icse_9
            left_stack_d[__idx_left_stack_1_7] = leftd
            left_stack[__idx_left_stack_1_7] = left
            __idx_right_stack_1_9 = (prefix_right_stack_1[(i_level - 1) + 1] + __icse_10) + __icse_9
            right_stack_d[__idx_right_stack_1_9] = rightd
            right_stack[__idx_right_stack_1_9] = right
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        __idx_tripcount_stack_1_10 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + (max(0, div(nu1 - 1, 1) + 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_10] = nc
        for j = 1:nc
            jf = j * 2
            __idx_f_stack_1_1 = prefix_f_stack_1[(i_level - 1) + 1] + ((j - 1) + 1)
            f_stack_d[__idx_f_stack_1_1] = fd[j, i_level + 1]
            f_stack[__idx_f_stack_1_1] = f[j, i_level + 1]
            fd[j, i_level + 1] = (0.25 * rd[jf - 1, i_level] + 0.5 * rd[jf, i_level]) + 0.25 * rd[jf + 1, i_level]
            f[j, i_level + 1] = 0.25 * r[jf - 1, i_level] + 0.5 * r[jf, i_level] + 0.25 * r[jf + 1, i_level]
        end
        __idx_tripcount_stack_1_13 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + ((max(0, div(nu1 - 1, 1) + 1) + 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_13] = nc
        for j = 1:nc
            __idx_u_stack_1_0 = (prefix_u_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            u_stack_d[__idx_u_stack_1_0] = ud[j, i_level + 1]
            u_stack[__idx_u_stack_1_0] = u[j, i_level + 1]
            ud[j, i_level + 1] = 0.0
            u[j, i_level + 1] = 0.0
        end
        nl = ncg
        __idx_hl_stack_1_17 = prefix_hl_stack_1[(i_level - 1) + 1] + 1
        hl_stack_d[__idx_hl_stack_1_17] = hld
        hl_stack[__idx_hl_stack_1_17] = hl
        hld = 2.0hld
        hl = hl * 2.0
    end
    __idx_hl2_stack_4 = __tot_hl2_stack_1 + 1
    hl2_stack_d[__idx_hl2_stack_4] = hl2d
    hl2_stack[__idx_hl2_stack_4] = hl2
    __hcse_3 = hl * hld
    hl2d = __hcse_3 + __hcse_3
    hl2 = hl * hl
    __idx_u_stack_7 = __tot_u_stack_1 + 1
    u_stack_d[__idx_u_stack_7] = ud[1, num_levels]
    u_stack[__idx_u_stack_7] = u[1, num_levels]
    __hcse_4 = f[1, num_levels]
    ud[1, num_levels] = (0.5__hcse_4) * hl2d + (0.5hl2) * fd[1, num_levels]
    u[1, num_levels] = 0.5 * hl2 * __hcse_4
    for i_level = num_levels - 1:-1:1
        nl = nl * 2
        __idx_hl_stack_2_1 = (__tot_hl_stack_1 + prefix_hl_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        hl_stack_d[__idx_hl_stack_2_1] = hld
        hl_stack[__idx_hl_stack_2_1] = hl
        hld = 0.5hld
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        __idx_hl2_stack_2_7 = ((__tot_hl2_stack_1 + 1) + prefix_hl2_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        hl2_stack_d[__idx_hl2_stack_2_7] = hl2d
        hl2_stack[__idx_hl2_stack_2_7] = hl2
        __hcse_5 = hl * hld
        hl2d = __hcse_5 + __hcse_5
        hl2 = hl * hl
        __idx_tripcount_stack_2_10 = (__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        tripcount_stack[__idx_tripcount_stack_2_10] = nc
        for j = 1:nc
            jf = j * 2
            __idx_u_stack_2_1 = ((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            __cse_11d = ud[jf, i_level]
            __cse_11 = u[jf, i_level]
            u_stack_d[__idx_u_stack_2_1] = __cse_11d
            u_stack[__idx_u_stack_2_1] = __cse_11
            ud[jf, i_level] = __cse_11d + ud[j, i_level + 1]
            u[jf, i_level] = __cse_11 + u[j, i_level + 1]
        end
        __idx_tripcount_stack_2_13 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1) + 1
        tripcount_stack[__idx_tripcount_stack_2_13] = nc
        for j = 1:nc + 1
            jf = j * 2 - 1
            cld = 0.0
            cl = 0.0
            if j > 1
                __idx_branch_stack_2_0 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 1
                cld = ud[j - 1, i_level + 1]
                cl = u[j - 1, i_level + 1]
            else
                __idx_branch_stack_2_0 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 0
            end
            crd = 0.0
            cr = 0.0
            if j <= nc
                __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 1
                crd = ud[j, i_level + 1]
                cr = u[j, i_level + 1]
            else
                __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 0
            end
            __icse_12 = (j - 1) + 1
            __idx_u_stack_2_5 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div(val_nc_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + __icse_12
            __cse_13d = ud[jf, i_level]
            __cse_13 = u[jf, i_level]
            u_stack_d[__idx_u_stack_2_5] = __cse_13d
            u_stack[__idx_u_stack_2_5] = __cse_13
            ud[jf, i_level] = __cse_13d + 0.5 * (cld + crd)
            u[jf, i_level] = __cse_13 + 0.5 * (cl + cr)
            __idx_cl_stack_2_8 = prefix_cl_stack_2[div(i_level - (num_levels - 1), -1) + 1] + __icse_12
            cl_stack_d[__idx_cl_stack_2_8] = cld
            cl_stack[__idx_cl_stack_2_8] = cl
            __idx_cr_stack_2_10 = prefix_cr_stack_2[div(i_level - (num_levels - 1), -1) + 1] + __icse_12
            cr_stack_d[__idx_cr_stack_2_10] = crd
            cr_stack[__idx_cr_stack_2_10] = cr
        end
        for i_k = 1:nu2
            __idx_tripcount_stack_2_0 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (1 + 1)) + ((i_k - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_2_0] = n
            for i_j = 1:n
                leftd = 0.0
                left = 0.0
                if i_j > 1
                    __icse_14 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (__icse_14 + __icse_14)) + (((i_k - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 1
                    leftd = ud[i_j - 1, i_level]
                    left = u[i_j - 1, i_level]
                else
                    __icse_15 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (__icse_15 + __icse_15)) + (((i_k - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 0
                end
                rightd = 0.0
                right = 0.0
                if i_j < n
                    __icse_16 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                    __icse_17 = div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((__icse_16 + __icse_16) + max(0, div(nu2 - 1, 1) + 1) * max(0, __icse_17))) + (((i_k - 1) * __icse_17 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 1
                    rightd = ud[i_j + 1, i_level]
                    right = u[i_j + 1, i_level]
                else
                    __icse_18 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                    __icse_19 = div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((__icse_18 + __icse_18) + max(0, div(nu2 - 1, 1) + 1) * max(0, __icse_19))) + (((i_k - 1) * __icse_19 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 0
                end
                __icse_20 = val_nc_2[div(i_level - (num_levels - 1), -1) + 1]
                __icse_21 = ((i_k - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1
                __idx_u_stack_2_4 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (max(0, div(__icse_20 - 1, 1) + 1) + max(0, div((__icse_20 + 1) - 1, 1) + 1))) + __icse_21
                u_stack_d[__idx_u_stack_2_4] = ud[i_j, i_level]
                u_stack[__idx_u_stack_2_4] = u[i_j, i_level]
                __hcse_6 = f[i_j, i_level]
                ud[i_j, i_level] = 0.5 * (((__hcse_6 * hl2d + hl2 * fd[i_j, i_level]) + leftd) + rightd)
                u[i_j, i_level] = 0.5 * (hl2 * __hcse_6 + left + right)
                __idx_left_stack_2_7 = (__tot_left_stack_1 + prefix_left_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + __icse_21
                left_stack_d[__idx_left_stack_2_7] = leftd
                left_stack[__idx_left_stack_2_7] = left
                __idx_right_stack_2_9 = (__tot_right_stack_1 + prefix_right_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + __icse_21
                right_stack_d[__idx_right_stack_2_9] = rightd
                right_stack[__idx_right_stack_2_9] = right
            end
        end
    end
    __ihcse_7 = (__tot_hl_stack_1 + __tot_hl_stack_2) + 1
    __idx_hl_stack_11 = __ihcse_7
    hl_stack_d[__idx_hl_stack_11] = hld
    hl_stack[__idx_hl_stack_11] = hl
    __ihcse_8 = ((__tot_hl2_stack_1 + 1) + __tot_hl2_stack_2) + 1
    __idx_hl2_stack_13 = __ihcse_8
    hl2_stack_d[__idx_hl2_stack_13] = hl2d
    hl2_stack[__idx_hl2_stack_13] = hl2
    __idx_hl_stack_0 = __ihcse_7
    hld = hl_stack_d[__idx_hl_stack_0]
    hl = hl_stack[__idx_hl_stack_0]
    __idx_hl2_stack_2 = __ihcse_8
    hl2d = hl2_stack_d[__idx_hl2_stack_2]
    hl2 = hl2_stack[__idx_hl2_stack_2]
    for i_level = 1:num_levels - 1
        for i_k = nu2:-1:1
            __idx_tripcount_stack_2_0 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (1 + 1)) + ((i_k - 1) + 1)
            n = tripcount_stack[__idx_tripcount_stack_2_0]
            for i_j = n:-1:1
                __icse_22 = div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1
                __icse_23 = ((i_k - 1) * __icse_22 + (i_j - 1)) + 1
                __idx_left_stack_2_0 = (__tot_left_stack_1 + prefix_left_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + __icse_23
                leftd = left_stack_d[__idx_left_stack_2_0]
                left = left_stack[__idx_left_stack_2_0]
                __idx_right_stack_2_2 = (__tot_right_stack_1 + prefix_right_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + __icse_23
                rightd = right_stack_d[__idx_right_stack_2_2]
                right = right_stack[__idx_right_stack_2_2]
                __icse_24 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                __idx_branch_stack_2_4 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((__icse_24 + __icse_24) + max(0, div(nu2 - 1, 1) + 1) * max(0, __icse_22))) + __icse_23
                __branch_pre_4 = branch_stack[__idx_branch_stack_2_4]
                rightd = 0.0
                right = 0.0
                if __branch_pre_4 == 1
                    rightd = ud[i_j + 1, i_level]
                    right = u[i_j + 1, i_level]
                else
                    rightd = 0.0
                    right = 0.0
                end
                __icse_25 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                __idx_branch_stack_2_8 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (__icse_25 + __icse_25)) + (((i_k - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                __branch_pre_2 = branch_stack[__idx_branch_stack_2_8]
                leftd = 0.0
                left = 0.0
                if __branch_pre_2 == 1
                    leftd = ud[i_j - 1, i_level]
                    left = u[i_j - 1, i_level]
                else
                    leftd = 0.0
                    left = 0.0
                end
                __icse_26 = val_nc_2[div(i_level - (num_levels - 1), -1) + 1]
                __idx_u_stack_2_0 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (max(0, div(__icse_26 - 1, 1) + 1) + max(0, div((__icse_26 + 1) - 1, 1) + 1))) + (((i_k - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                ud[i_j, i_level] = u_stack_d[__idx_u_stack_2_0]
                u[i_j, i_level] = u_stack[__idx_u_stack_2_0]
                __oldb_2d = ubd[i_j, i_level]
                __oldb_2 = ub[i_j, i_level]
                ubd[i_j, i_level] = 0.0
                ub[i_j, i_level] = 0.0
                __cse_27d = 0.5__oldb_2d
                __cse_27 = 0.5__oldb_2
                __hcse_9 = f[i_j, i_level]
                hl2bd = hl2bd + (__cse_27 * fd[i_j, i_level] + __hcse_9 * __cse_27d)
                hl2b = hl2b + __hcse_9 * __cse_27
                fbd[i_j, i_level] = fbd[i_j, i_level] + (__cse_27 * hl2d + hl2 * __cse_27d)
                fb[i_j, i_level] = fb[i_j, i_level] + hl2 * __cse_27
                leftbd = leftbd + __cse_27d
                leftb = leftb + __cse_27
                rightbd = rightbd + __cse_27d
                rightb = rightb + __cse_27
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
        __idx_tripcount_stack_2_1 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1) + 1
        nc = tripcount_stack[__idx_tripcount_stack_2_1]
        for j = nc + 1:-1:1
            __icse_28 = (j - 1) + 1
            __idx_cl_stack_2_0 = prefix_cl_stack_2[div(i_level - (num_levels - 1), -1) + 1] + __icse_28
            cld = cl_stack_d[__idx_cl_stack_2_0]
            cl = cl_stack[__idx_cl_stack_2_0]
            __idx_cr_stack_2_2 = prefix_cr_stack_2[div(i_level - (num_levels - 1), -1) + 1] + __icse_28
            crd = cr_stack_d[__idx_cr_stack_2_2]
            cr = cr_stack[__idx_cr_stack_2_2]
            jf = j * 2 - 1
            __idx_branch_stack_2_5 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + __icse_28
            __branch_pre_5 = branch_stack[__idx_branch_stack_2_5]
            crd = 0.0
            cr = 0.0
            if __branch_pre_5 == 1
                crd = ud[j, i_level + 1]
                cr = u[j, i_level + 1]
            else
                crd = 0.0
                cr = 0.0
            end
            __idx_branch_stack_2_9 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            __branch_pre_3 = branch_stack[__idx_branch_stack_2_9]
            cld = 0.0
            cl = 0.0
            if __branch_pre_3 == 1
                cld = ud[j - 1, i_level + 1]
                cl = u[j - 1, i_level + 1]
            else
                cld = 0.0
                cl = 0.0
            end
            __idx_u_stack_2_0 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div(val_nc_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            ud[jf, i_level] = u_stack_d[__idx_u_stack_2_0]
            u[jf, i_level] = u_stack[__idx_u_stack_2_0]
            __cse_29d = 0.5 * ubd[jf, i_level]
            __cse_29 = 0.5 * ub[jf, i_level]
            clbd = clbd + __cse_29d
            clb = clb + __cse_29
            crbd = crbd + __cse_29d
            crb = crb + __cse_29
            if __branch_pre_5 == 1
                __oldb_0d = crbd
                __oldb_0 = crb
                crbd = 0.0
                crb = 0.0
                ubd[j, i_level + 1] = ubd[j, i_level + 1] + __oldb_0d
                ub[j, i_level + 1] = ub[j, i_level + 1] + __oldb_0
            end
            crbd = 0.0
            crb = 0.0
            if __branch_pre_3 == 1
                __oldb_0d = clbd
                __oldb_0 = clb
                clbd = 0.0
                clb = 0.0
                ubd[j - 1, i_level + 1] = ubd[j - 1, i_level + 1] + __oldb_0d
                ub[j - 1, i_level + 1] = ub[j - 1, i_level + 1] + __oldb_0
            end
            clbd = 0.0
            clb = 0.0
        end
        __idx_tripcount_stack_2_4 = (__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        nc = tripcount_stack[__idx_tripcount_stack_2_4]
        for j = nc:-1:1
            jf = j * 2
            __idx_u_stack_2_0 = ((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            ud[jf, i_level] = u_stack_d[__idx_u_stack_2_0]
            u[jf, i_level] = u_stack[__idx_u_stack_2_0]
            ubd[j, i_level + 1] = ubd[j, i_level + 1] + ubd[jf, i_level]
            ub[j, i_level + 1] = ub[j, i_level + 1] + ub[jf, i_level]
        end
        __idx_hl2_stack_2_0 = ((__tot_hl2_stack_1 + 1) + prefix_hl2_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        hl2d = hl2_stack_d[__idx_hl2_stack_2_0]
        hl2 = hl2_stack[__idx_hl2_stack_2_0]
        __oldb_2d = hl2bd
        __oldb_2 = hl2b
        hl2bd = 0.0
        hl2b = 0.0
        __cse_30d = __oldb_2 * hld + hl * __oldb_2d
        __cse_30 = hl * __oldb_2
        hlbd = hlbd + __cse_30d
        hlb = hlb + __cse_30
        hlbd = hlbd + __cse_30d
        hlb = hlb + __cse_30
        __idx_hl_stack_2_0 = (__tot_hl_stack_1 + prefix_hl_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        hld = hl_stack_d[__idx_hl_stack_2_0]
        hl = hl_stack[__idx_hl_stack_2_0]
        __oldb_2d = hlbd
        __oldb_2 = hlb
        hlbd = 0.0
        hlb = 0.0
        hlbd = hlbd + 0.5__oldb_2d
        hlb = hlb + 0.5__oldb_2
    end
    __idx_u_stack_0 = __tot_u_stack_1 + 1
    ud[1, num_levels] = u_stack_d[__idx_u_stack_0]
    u[1, num_levels] = u_stack[__idx_u_stack_0]
    __oldb_2d = ubd[1, num_levels]
    __oldb_2 = ub[1, num_levels]
    ubd[1, num_levels] = 0.0
    ub[1, num_levels] = 0.0
    __hcse_10 = 0.5 * f[1, num_levels]
    hl2bd = hl2bd + (__oldb_2 * (0.5 * fd[1, num_levels]) + __hcse_10 * __oldb_2d)
    hl2b = hl2b + __hcse_10 * __oldb_2
    __hcse_11 = 0.5hl2
    fbd[1, num_levels] = fbd[1, num_levels] + (__oldb_2 * (0.5hl2d) + __hcse_11 * __oldb_2d)
    fb[1, num_levels] = fb[1, num_levels] + __hcse_11 * __oldb_2
    __idx_hl2_stack_0 = __tot_hl2_stack_1 + 1
    hl2d = hl2_stack_d[__idx_hl2_stack_0]
    hl2 = hl2_stack[__idx_hl2_stack_0]
    __oldb_2d = hl2bd
    __oldb_2 = hl2b
    hl2bd = 0.0
    hl2b = 0.0
    __cse_31d = __oldb_2 * hld + hl * __oldb_2d
    __cse_31 = hl * __oldb_2
    hlbd = hlbd + __cse_31d
    hlb = hlb + __cse_31
    hlbd = hlbd + __cse_31d
    hlb = hlb + __cse_31
    for i_level = num_levels - 1:-1:1
        __idx_hl_stack_1_0 = prefix_hl_stack_1[(i_level - 1) + 1] + 1
        hld = hl_stack_d[__idx_hl_stack_1_0]
        hl = hl_stack[__idx_hl_stack_1_0]
        __oldb_2d = hlbd
        __oldb_2 = hlb
        hlbd = 0.0
        hlb = 0.0
        hlbd = hlbd + 2.0__oldb_2d
        hlb = hlb + 2.0__oldb_2
        __idx_tripcount_stack_1_5 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + ((max(0, div(nu1 - 1, 1) + 1) + 1) + 1)) + 1
        nc = tripcount_stack[__idx_tripcount_stack_1_5]
        for j = nc:-1:1
            __idx_u_stack_1_0 = (prefix_u_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            ud[j, i_level + 1] = u_stack_d[__idx_u_stack_1_0]
            u[j, i_level + 1] = u_stack[__idx_u_stack_1_0]
            ubd[j, i_level + 1] = 0.0
            ub[j, i_level + 1] = 0.0
        end
        __idx_tripcount_stack_1_8 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + (max(0, div(nu1 - 1, 1) + 1) + 1)) + 1
        nc = tripcount_stack[__idx_tripcount_stack_1_8]
        for j = nc:-1:1
            jf = j * 2
            __idx_f_stack_1_0 = prefix_f_stack_1[(i_level - 1) + 1] + ((j - 1) + 1)
            fd[j, i_level + 1] = f_stack_d[__idx_f_stack_1_0]
            f[j, i_level + 1] = f_stack[__idx_f_stack_1_0]
            __oldb_2d = fbd[j, i_level + 1]
            __oldb_2 = fb[j, i_level + 1]
            fbd[j, i_level + 1] = 0.0
            fb[j, i_level + 1] = 0.0
            __cse_32d = 0.25__oldb_2d
            __cse_32 = 0.25__oldb_2
            rbd[jf - 1, i_level] = rbd[jf - 1, i_level] + __cse_32d
            rb[jf - 1, i_level] = rb[jf - 1, i_level] + __cse_32
            rbd[jf, i_level] = rbd[jf, i_level] + 0.5__oldb_2d
            rb[jf, i_level] = rb[jf, i_level] + 0.5__oldb_2
            rbd[jf + 1, i_level] = rbd[jf + 1, i_level] + __cse_32d
            rb[jf + 1, i_level] = rb[jf + 1, i_level] + __cse_32
        end
        __idx_tripcount_stack_1_11 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1)) + 1
        n = tripcount_stack[__idx_tripcount_stack_1_11]
        for j = n:-1:1
            __icse_33 = max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
            __icse_34 = max(0, div(nu1 - 1, 1) + 1) * __icse_33
            __icse_35 = (j - 1) + 1
            __idx_left_stack_1_0 = (prefix_left_stack_1[(i_level - 1) + 1] + __icse_34) + __icse_35
            leftd = left_stack_d[__idx_left_stack_1_0]
            left = left_stack[__idx_left_stack_1_0]
            __idx_right_stack_1_2 = (prefix_right_stack_1[(i_level - 1) + 1] + __icse_34) + __icse_35
            rightd = right_stack_d[__idx_right_stack_1_2]
            right = right_stack[__idx_right_stack_1_2]
            __idx_branch_stack_1_4 = (prefix_branch_stack_1[(i_level - 1) + 1] + ((__icse_34 + __icse_34) + __icse_33)) + __icse_35
            __branch_pre_4 = branch_stack[__idx_branch_stack_1_4]
            rightd = 0.0
            right = 0.0
            if __branch_pre_4 == 1
                rightd = ud[j + 1, i_level]
                right = u[j + 1, i_level]
            else
                rightd = 0.0
                right = 0.0
            end
            __icse_36 = max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
            __idx_branch_stack_1_8 = (prefix_branch_stack_1[(i_level - 1) + 1] + (__icse_36 + __icse_36)) + ((j - 1) + 1)
            __branch_pre_2 = branch_stack[__idx_branch_stack_1_8]
            leftd = 0.0
            left = 0.0
            if __branch_pre_2 == 1
                leftd = ud[j - 1, i_level]
                left = u[j - 1, i_level]
            else
                leftd = 0.0
                left = 0.0
            end
            __idx_r_stack_1_0 = prefix_r_stack_1[(i_level - 1) + 1] + ((j - 1) + 1)
            rd[j, i_level] = r_stack_d[__idx_r_stack_1_0]
            r[j, i_level] = r_stack[__idx_r_stack_1_0]
            __oldb_2d = rbd[j, i_level]
            __oldb_2 = rb[j, i_level]
            rbd[j, i_level] = 0.0
            rb[j, i_level] = 0.0
            fbd[j, i_level] = fbd[j, i_level] + __oldb_2d
            fb[j, i_level] = fb[j, i_level] + __oldb_2
            __cse_37d = -__oldb_2d
            __cse_37 = -__oldb_2
            __hcse_12 = hl2 ^ 2
            __hcse_13 = 1.0 / __hcse_12
            __hcse_14 = 1.0 / hl2
            __cse_38d = __cse_37 * (-__hcse_13 * hl2d) + __hcse_14 * __cse_37d
            __cse_38 = __hcse_14 * __cse_37
            ubd[j, i_level] = ubd[j, i_level] + 2.0__cse_38d
            ub[j, i_level] = ub[j, i_level] + 2.0__cse_38
            __cse_39d = -__cse_38d
            __cse_39 = -__cse_38
            leftbd = leftbd + __cse_39d
            leftb = leftb + __cse_39
            rightbd = rightbd + __cse_39d
            rightb = rightb + __cse_39
            __hcse_15 = (2.0 * u[j, i_level] - left) - right
            __hcse_16 = -(__hcse_15 / __hcse_12)
            hl2bd = hl2bd + (__cse_37 * -((__hcse_13 * ((2.0 * ud[j, i_level] + -leftd) + -rightd) + -(__hcse_15 / __hcse_12 ^ 2) * ((2hl2) * hl2d))) + __hcse_16 * __cse_37d)
            hl2b = hl2b + __hcse_16 * __cse_37
            if __branch_pre_4 == 1
                __oldb_0d = rightbd
                __oldb_0 = rightb
                rightbd = 0.0
                rightb = 0.0
                ubd[j + 1, i_level] = ubd[j + 1, i_level] + __oldb_0d
                ub[j + 1, i_level] = ub[j + 1, i_level] + __oldb_0
            end
            rightbd = 0.0
            rightb = 0.0
            if __branch_pre_2 == 1
                __oldb_0d = leftbd
                __oldb_0 = leftb
                leftbd = 0.0
                leftb = 0.0
                ubd[j - 1, i_level] = ubd[j - 1, i_level] + __oldb_0d
                ub[j - 1, i_level] = ub[j - 1, i_level] + __oldb_0
            end
            leftbd = 0.0
            leftb = 0.0
        end
        for i_k = nu1:-1:1
            __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_level - 1) + 1] + ((i_k - 1) + 1)
            n = tripcount_stack[__idx_tripcount_stack_1_0]
            for i_j = n:-1:1
                __icse_40 = div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1
                __icse_41 = ((i_k - 1) * __icse_40 + (i_j - 1)) + 1
                __idx_left_stack_1_0 = prefix_left_stack_1[(i_level - 1) + 1] + __icse_41
                leftd = left_stack_d[__idx_left_stack_1_0]
                left = left_stack[__idx_left_stack_1_0]
                __idx_right_stack_1_2 = prefix_right_stack_1[(i_level - 1) + 1] + __icse_41
                rightd = right_stack_d[__idx_right_stack_1_2]
                right = right_stack[__idx_right_stack_1_2]
                __idx_branch_stack_1_4 = (prefix_branch_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, __icse_40)) + __icse_41
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
                __idx_branch_stack_1_8 = prefix_branch_stack_1[(i_level - 1) + 1] + (((i_k - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
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
                __idx_u_stack_1_0 = prefix_u_stack_1[(i_level - 1) + 1] + (((i_k - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                ud[i_j, i_level] = u_stack_d[__idx_u_stack_1_0]
                u[i_j, i_level] = u_stack[__idx_u_stack_1_0]
                __oldb_2d = ubd[i_j, i_level]
                __oldb_2 = ub[i_j, i_level]
                ubd[i_j, i_level] = 0.0
                ub[i_j, i_level] = 0.0
                __cse_42d = 0.5__oldb_2d
                __cse_42 = 0.5__oldb_2
                __hcse_17 = f[i_j, i_level]
                hl2bd = hl2bd + (__cse_42 * fd[i_j, i_level] + __hcse_17 * __cse_42d)
                hl2b = hl2b + __hcse_17 * __cse_42
                fbd[i_j, i_level] = fbd[i_j, i_level] + (__cse_42 * hl2d + hl2 * __cse_42d)
                fb[i_j, i_level] = fb[i_j, i_level] + hl2 * __cse_42
                leftbd = leftbd + __cse_42d
                leftb = leftb + __cse_42
                rightbd = rightbd + __cse_42d
                rightb = rightb + __cse_42
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
        __idx_hl2_stack_1_0 = prefix_hl2_stack_1[(i_level - 1) + 1] + 1
        hl2d = hl2_stack_d[__idx_hl2_stack_1_0]
        hl2 = hl2_stack[__idx_hl2_stack_1_0]
        __oldb_2d = hl2bd
        __oldb_2 = hl2b
        hl2bd = 0.0
        hl2b = 0.0
        __cse_43d = __oldb_2 * hld + hl * __oldb_2d
        __cse_43 = hl * __oldb_2
        hlbd = hlbd + __cse_43d
        hlb = hlb + __cse_43
        hlbd = hlbd + __cse_43d
        hlb = hlb + __cse_43
    end
    __oldb_0d = hlbd
    __oldb_0 = hlb
    hlbd = 0.0
    hlb = 0.0
    h1bd = h1bd + __oldb_0d
    h1b = h1b + __oldb_0
    return (h1b, h1bd)
end

function mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    n = n * 2
    nl = nfine
    hl = h1
    for i_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_k = 1:nu1
            for i_j = 1:n
                left = 0.0
                if i_j > 1
                    left = u[i_j - 1, i_level]
                end
                right = 0.0
                if i_j < n
                    right = u[i_j + 1, i_level]
                end
                u[i_j, i_level] = 0.5 * (hl2 * f[i_j, i_level] + left + right)
            end
        end
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_level]
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_level]
            end
            r[j, i_level] = f[j, i_level] - ((2.0 * u[j, i_level] - left) - right) / hl2
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        for j = 1:nc
            jf = j * 2
            f[j, i_level + 1] = 0.25 * r[jf - 1, i_level] + 0.5 * r[jf, i_level] + 0.25 * r[jf + 1, i_level]
        end
        for j = 1:nc
            u[j, i_level + 1] = 0.0
        end
        nl = ncg
        hl = hl * 2.0
    end
    hl2 = hl * hl
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_level = num_levels - 1:-1:1
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        for j = 1:nc
            jf = j * 2
            u[jf, i_level] = u[jf, i_level] + u[j, i_level + 1]
        end
        for j = 1:nc + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level + 1]
            end
            cr = 0.0
            if j <= nc
                cr = u[j, i_level + 1]
            end
            u[jf, i_level] = u[jf, i_level] + 0.5 * (cl + cr)
        end
        for i_k = 1:nu2
            for i_j = 1:n
                left = 0.0
                if i_j > 1
                    left = u[i_j - 1, i_level]
                end
                right = 0.0
                if i_j < n
                    right = u[i_j + 1, i_level]
                end
                u[i_j, i_level] = 0.5 * (hl2 * f[i_j, i_level] + left + right)
            end
        end
    end
end
