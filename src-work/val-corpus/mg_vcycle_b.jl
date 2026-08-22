function initstacks_mg_vcycle_b(h1, n, nfine, nu1, nu2, num_levels)
    n = n * 2
    nl = nfine
    hl = h1
    prefix_branch_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_branch_stack_1 = 0
    prefix_f_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_f_stack_1 = 0
    prefix_hl2_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_hl2_stack_1 = 0
    prefix_hl_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_hl_stack_1 = 0
    prefix_left_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_left_stack_1 = 0
    prefix_r_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_r_stack_1 = 0
    prefix_right_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_right_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_tripcount_stack_1 = 0
    prefix_u_stack_1 = Vector{Int}(undef, div((num_levels - 1) - 1, 1) + 1)
    __tot_u_stack_1 = 0
    val_n_1 = Vector{Int64}(undef, div((num_levels - 1) - 1, 1) + 1)
    val_nc_1 = Vector{Int64}(undef, div((num_levels - 1) - 1, 1) + 1)
    for i_seq_level = 1:num_levels - 1
        prefix_branch_stack_1[(i_seq_level - 1) + 1] = __tot_branch_stack_1
        prefix_f_stack_1[(i_seq_level - 1) + 1] = __tot_f_stack_1
        prefix_hl2_stack_1[(i_seq_level - 1) + 1] = __tot_hl2_stack_1
        prefix_hl_stack_1[(i_seq_level - 1) + 1] = __tot_hl_stack_1
        prefix_left_stack_1[(i_seq_level - 1) + 1] = __tot_left_stack_1
        prefix_r_stack_1[(i_seq_level - 1) + 1] = __tot_r_stack_1
        prefix_right_stack_1[(i_seq_level - 1) + 1] = __tot_right_stack_1
        prefix_tripcount_stack_1[(i_seq_level - 1) + 1] = __tot_tripcount_stack_1
        prefix_u_stack_1[(i_seq_level - 1) + 1] = __tot_u_stack_1
        n = nl - 1
        hl2 = hl * hl
        ncg = div(nl, 2)
        nc = ncg - 1
        nl = ncg
        hl = hl * 2.0
        val_n_1[(i_seq_level - 1) + 1] = n
        val_nc_1[(i_seq_level - 1) + 1] = nc
        __tot_branch_stack_1 = __tot_branch_stack_1 + ((((div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(n - 1, 1) + 1)) + (div(n - 1, 1) + 1))
        __tot_f_stack_1 = __tot_f_stack_1 + (div(nc - 1, 1) + 1)
        __tot_hl2_stack_1 = __tot_hl2_stack_1 + 1
        __tot_hl_stack_1 = __tot_hl_stack_1 + 1
        __tot_left_stack_1 = __tot_left_stack_1 + ((((((((div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(n - 1, 1) + 1)) + (div(n - 1, 1) + 1)) + (div(n - 1, 1) + 1)) + 1)
        __tot_r_stack_1 = __tot_r_stack_1 + (div(n - 1, 1) + 1)
        __tot_right_stack_1 = __tot_right_stack_1 + ((((((((div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(n - 1, 1) + 1)) + (div(n - 1, 1) + 1)) + (div(n - 1, 1) + 1)) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + ((((div(nu1 - 1, 1) + 1) + 1) + 1) + 1)
        __tot_u_stack_1 = __tot_u_stack_1 + ((div(nu1 - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(nc - 1, 1) + 1))
    end
    hl2 = hl * hl
    prefix_branch_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_branch_stack_2 = 0
    prefix_cl_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_cl_stack_2 = 0
    prefix_cr_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_cr_stack_2 = 0
    prefix_hl2_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_hl2_stack_2 = 0
    prefix_hl_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_hl_stack_2 = 0
    prefix_left_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_left_stack_2 = 0
    prefix_right_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_right_stack_2 = 0
    prefix_tripcount_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_tripcount_stack_2 = 0
    prefix_u_stack_2 = Vector{Int}(undef, div(1 - (num_levels - 1), -1) + 1)
    __tot_u_stack_2 = 0
    val_n_2 = Vector{Int64}(undef, div(1 - (num_levels - 1), -1) + 1)
    val_nc_2 = Vector{Int64}(undef, div(1 - (num_levels - 1), -1) + 1)
    for i_seq_level = num_levels - 1:-1:1
        prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_branch_stack_2
        prefix_cl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_cl_stack_2
        prefix_cr_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_cr_stack_2
        prefix_hl2_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_hl2_stack_2
        prefix_hl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_hl_stack_2
        prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_left_stack_2
        prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_right_stack_2
        prefix_tripcount_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_tripcount_stack_2
        prefix_u_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] = __tot_u_stack_2
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] = n
        val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] = nc
        __tot_branch_stack_2 = __tot_branch_stack_2 + ((((div((nc + 1) - 1, 1) + 1) + (div((nc + 1) - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1))
        __tot_cl_stack_2 = __tot_cl_stack_2 + ((((div((nc + 1) - 1, 1) + 1) + (div((nc + 1) - 1, 1) + 1)) + (div((nc + 1) - 1, 1) + 1)) + 1)
        __tot_cr_stack_2 = __tot_cr_stack_2 + ((((div((nc + 1) - 1, 1) + 1) + (div((nc + 1) - 1, 1) + 1)) + (div((nc + 1) - 1, 1) + 1)) + 1)
        __tot_hl2_stack_2 = __tot_hl2_stack_2 + 1
        __tot_hl_stack_2 = __tot_hl_stack_2 + 1
        __tot_left_stack_2 = __tot_left_stack_2 + (((((div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1)) + 1)
        __tot_right_stack_2 = __tot_right_stack_2 + (((((div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1)) + 1)
        __tot_tripcount_stack_2 = __tot_tripcount_stack_2 + ((1 + 1) + (div(nu2 - 1, 1) + 1))
        __tot_u_stack_2 = __tot_u_stack_2 + (((div(nc - 1, 1) + 1) + (div((nc + 1) - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(n - 1, 1) + 1))
    end
    hl2_stack = Vector{Float64}(undef, ((__tot_hl2_stack_1 + 1) + __tot_hl2_stack_2) + 1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1 + __tot_tripcount_stack_2)
    left_stack = Vector{Float64}(undef, (__tot_left_stack_1 + __tot_left_stack_2) + 1)
    branch_stack = Vector{Int64}(undef, __tot_branch_stack_1 + __tot_branch_stack_2)
    right_stack = Vector{Float64}(undef, (__tot_right_stack_1 + __tot_right_stack_2) + 1)
    u_stack = Vector{Float64}(undef, (__tot_u_stack_1 + 1) + __tot_u_stack_2)
    r_stack = Vector{Float64}(undef, __tot_r_stack_1)
    f_stack = Vector{Float64}(undef, __tot_f_stack_1)
    hl_stack = Vector{Float64}(undef, (__tot_hl_stack_1 + __tot_hl_stack_2) + 1)
    cl_stack = Vector{Float64}(undef, __tot_cl_stack_2 + 1)
    cr_stack = Vector{Float64}(undef, __tot_cr_stack_2 + 1)
    return (hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, u_stack, r_stack, f_stack, hl_stack, cl_stack, cr_stack, prefix_branch_stack_1, prefix_f_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_r_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, prefix_u_stack_1, prefix_branch_stack_2, prefix_cl_stack_2, prefix_cr_stack_2, prefix_hl2_stack_2, prefix_hl_stack_2, prefix_left_stack_2, prefix_right_stack_2, prefix_tripcount_stack_2, prefix_u_stack_2, __tot_branch_stack_1, __tot_f_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_r_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, __tot_branch_stack_2, __tot_cl_stack_2, __tot_cr_stack_2, __tot_hl2_stack_2, __tot_hl_stack_2, __tot_left_stack_2, __tot_right_stack_2, __tot_tripcount_stack_2, __tot_u_stack_2, val_n_1, val_nc_1, val_n_2, val_nc_2)
end

function mg_vcycle_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, hl2_stack, tripcount_stack, left_stack, branch_stack, right_stack, u_stack, r_stack, f_stack, hl_stack, cl_stack, cr_stack, prefix_branch_stack_1, prefix_f_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_r_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, prefix_u_stack_1, prefix_branch_stack_2, prefix_cl_stack_2, prefix_cr_stack_2, prefix_hl2_stack_2, prefix_hl_stack_2, prefix_left_stack_2, prefix_right_stack_2, prefix_tripcount_stack_2, prefix_u_stack_2, __tot_branch_stack_1, __tot_f_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_r_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, __tot_branch_stack_2, __tot_cl_stack_2, __tot_cr_stack_2, __tot_hl2_stack_2, __tot_hl_stack_2, __tot_left_stack_2, __tot_right_stack_2, __tot_tripcount_stack_2, __tot_u_stack_2, val_n_1, val_nc_1, val_n_2, val_nc_2)
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
    n = n * 2
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        __idx_hl2_stack_1_1 = prefix_hl2_stack_1[(i_seq_level - 1) + 1] + 1
        hl2_stack[__idx_hl2_stack_1_1] = hl2
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_seq_level - 1) + 1] + ((i_seq_k - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_0] = n
            for i_seq_j = 1:n
                __idx_left_stack_1_0 = prefix_left_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                left_stack[__idx_left_stack_1_0] = left
                left = 0.0
                if i_seq_j > 1
                    __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    __idx_left_stack_1_0 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    left_stack[__idx_left_stack_1_0] = left
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                __idx_right_stack_1_4 = prefix_right_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                right_stack[__idx_right_stack_1_4] = right
                right = 0.0
                if i_seq_j < n
                    __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    __idx_right_stack_1_0 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    right_stack[__idx_right_stack_1_0] = right
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                __idx_u_stack_1_8 = prefix_u_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                u_stack[__idx_u_stack_1_8] = u[i_seq_j, i_seq_level]
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
                __idx_left_stack_1_11 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                left_stack[__idx_left_stack_1_11] = left
                __idx_right_stack_1_13 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                right_stack[__idx_right_stack_1_13] = right
            end
            __idx_left_stack_1_3 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)
            left_stack[__idx_left_stack_1_3] = left
            __idx_right_stack_1_5 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)
            right_stack[__idx_right_stack_1_5] = right
        end
        __idx_tripcount_stack_1_5 = (prefix_tripcount_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_5] = n
        for j = 1:n
            __idx_left_stack_1_0 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + ((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1))) + ((j - 1) + 1)
            left_stack[__idx_left_stack_1_0] = left
            left = 0.0
            if j > 1
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                __idx_left_stack_1_0 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + (((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
                left_stack[__idx_left_stack_1_0] = left
                left = u[j - 1, i_seq_level]
            else
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            __idx_right_stack_1_4 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + ((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1))) + ((j - 1) + 1)
            right_stack[__idx_right_stack_1_4] = right
            right = 0.0
            if j < n
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                __idx_right_stack_1_0 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + (((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
                right_stack[__idx_right_stack_1_0] = right
                right = u[j + 1, i_seq_level]
            else
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            __idx_r_stack_1_8 = prefix_r_stack_1[(i_seq_level - 1) + 1] + ((j - 1) + 1)
            r_stack[__idx_r_stack_1_8] = r[j, i_seq_level]
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
            __idx_left_stack_1_11 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + ((((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
            left_stack[__idx_left_stack_1_11] = left
            __idx_right_stack_1_13 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + ((((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
            right_stack[__idx_right_stack_1_13] = right
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        __idx_tripcount_stack_1_10 = (prefix_tripcount_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_10] = nc
        for j = 1:nc
            jf = j * 2
            __idx_f_stack_1_1 = prefix_f_stack_1[(i_seq_level - 1) + 1] + ((j - 1) + 1)
            f_stack[__idx_f_stack_1_1] = f[j, i_seq_level + 1]
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        __idx_tripcount_stack_1_13 = (prefix_tripcount_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) + 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_13] = nc
        for j = 1:nc
            __idx_u_stack_1_0 = (prefix_u_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            u_stack[__idx_u_stack_1_0] = u[j, i_seq_level + 1]
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        __idx_hl_stack_1_17 = prefix_hl_stack_1[(i_seq_level - 1) + 1] + 1
        hl_stack[__idx_hl_stack_1_17] = hl
        hl = hl * 2.0
        __idx_left_stack_1_20 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + (((((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + 1
        left_stack[__idx_left_stack_1_20] = left
        __idx_right_stack_1_22 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + (((((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + 1
        right_stack[__idx_right_stack_1_22] = right
    end
    hl2_stack[__tot_hl2_stack_1 + 1] = hl2
    hl2 = hl * hl
    u_stack[__tot_u_stack_1 + 1] = u[1, num_levels]
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        __idx_hl_stack_2_1 = (__tot_hl_stack_1 + prefix_hl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + 1
        hl_stack[__idx_hl_stack_2_1] = hl
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        __idx_hl2_stack_2_7 = ((__tot_hl2_stack_1 + 1) + prefix_hl2_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + 1
        hl2_stack[__idx_hl2_stack_2_7] = hl2
        hl2 = hl * hl
        __idx_tripcount_stack_2_10 = (__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + 1
        tripcount_stack[__idx_tripcount_stack_2_10] = nc
        for j = 1:nc
            jf = j * 2
            __idx_u_stack_2_1 = ((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            u_stack[__idx_u_stack_2_1] = u[jf, i_seq_level]
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        __idx_tripcount_stack_2_13 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + 1) + 1
        tripcount_stack[__idx_tripcount_stack_2_13] = nc
        for j = 1:nc + 1
            jf = j * 2 - 1
            __idx_cl_stack_2_1 = prefix_cl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + ((j - 1) + 1)
            cl_stack[__idx_cl_stack_2_1] = cl
            cl = 0.0
            if j > 1
                __idx_branch_stack_2_0 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 1
                __idx_cl_stack_2_0 = (prefix_cl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                cl_stack[__idx_cl_stack_2_0] = cl
                cl = u[j - 1, i_seq_level + 1]
            else
                __idx_branch_stack_2_0 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 0
            end
            __idx_cr_stack_2_5 = prefix_cr_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + ((j - 1) + 1)
            cr_stack[__idx_cr_stack_2_5] = cr
            cr = 0.0
            if j <= nc
                __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 1
                __idx_cr_stack_2_0 = (prefix_cr_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                cr_stack[__idx_cr_stack_2_0] = cr
                cr = u[j, i_seq_level + 1]
            else
                __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 0
            end
            __idx_u_stack_2_9 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (div(val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            u_stack[__idx_u_stack_2_9] = u[jf, i_seq_level]
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
            __idx_cl_stack_2_12 = (prefix_cl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + ((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)
            cl_stack[__idx_cl_stack_2_12] = cl
            __idx_cr_stack_2_14 = (prefix_cr_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + ((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)
            cr_stack[__idx_cr_stack_2_14] = cr
        end
        for i_seq_k = 1:nu2
            __idx_tripcount_stack_2_0 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (1 + 1)) + ((i_seq_k - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_2_0] = n
            for i_seq_j = 1:n
                __idx_left_stack_2_0 = (__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                left_stack[__idx_left_stack_2_0] = left
                left = 0.0
                if i_seq_j > 1
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 1
                    __idx_left_stack_2_0 = ((__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    left_stack[__idx_left_stack_2_0] = left
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 0
                end
                __idx_right_stack_2_4 = (__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                right_stack[__idx_right_stack_2_4] = right
                right = 0.0
                if i_seq_j < n
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 1
                    __idx_right_stack_2_0 = ((__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    right_stack[__idx_right_stack_2_0] = right
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 0
                end
                __idx_u_stack_2_8 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div(val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                u_stack[__idx_u_stack_2_8] = u[i_seq_j, i_seq_level]
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
                __idx_left_stack_2_11 = ((__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                left_stack[__idx_left_stack_2_11] = left
                __idx_right_stack_2_13 = ((__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                right_stack[__idx_right_stack_2_13] = right
            end
            __idx_left_stack_2_3 = ((__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)
            left_stack[__idx_left_stack_2_3] = left
            __idx_right_stack_2_5 = ((__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)
            right_stack[__idx_right_stack_2_5] = right
        end
        __idx_cl_stack_2_17 = (prefix_cl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + (((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + 1
        cl_stack[__idx_cl_stack_2_17] = cl
        __idx_cr_stack_2_19 = (prefix_cr_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + (((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + 1
        cr_stack[__idx_cr_stack_2_19] = cr
        __idx_left_stack_2_21 = ((__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1))) + 1
        left_stack[__idx_left_stack_2_21] = left
        __idx_right_stack_2_23 = ((__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1))) + 1
        right_stack[__idx_right_stack_2_23] = right
    end
    cl_stack[__tot_cl_stack_2 + 1] = cl
    cr_stack[__tot_cr_stack_2 + 1] = cr
    hl_stack[(__tot_hl_stack_1 + __tot_hl_stack_2) + 1] = hl
    hl2_stack[((__tot_hl2_stack_1 + 1) + __tot_hl2_stack_2) + 1] = hl2
    left_stack[(__tot_left_stack_1 + __tot_left_stack_2) + 1] = left
    right_stack[(__tot_right_stack_1 + __tot_right_stack_2) + 1] = right
    cl = cl_stack[__tot_cl_stack_2 + 1]
    cr = cr_stack[__tot_cr_stack_2 + 1]
    hl = hl_stack[(__tot_hl_stack_1 + __tot_hl_stack_2) + 1]
    hl2 = hl2_stack[((__tot_hl2_stack_1 + 1) + __tot_hl2_stack_2) + 1]
    left = left_stack[(__tot_left_stack_1 + __tot_left_stack_2) + 1]
    right = right_stack[(__tot_right_stack_1 + __tot_right_stack_2) + 1]
    for i_seq_level = 1:num_levels - 1
        __idx_cl_stack_2_0 = (prefix_cl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + (((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + 1
        cl = cl_stack[__idx_cl_stack_2_0]
        __idx_cr_stack_2_2 = (prefix_cr_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + (((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + 1
        cr = cr_stack[__idx_cr_stack_2_2]
        __idx_left_stack_2_4 = ((__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1))) + 1
        left = left_stack[__idx_left_stack_2_4]
        __idx_right_stack_2_6 = ((__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1))) + 1
        right = right_stack[__idx_right_stack_2_6]
        for i_seq_k = nu2:-1:1
            __idx_left_stack_2_0 = ((__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)
            left = left_stack[__idx_left_stack_2_0]
            __idx_right_stack_2_2 = ((__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)
            right = right_stack[__idx_right_stack_2_2]
            __idx_tripcount_stack_2_4 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (1 + 1)) + ((i_seq_k - 1) + 1)
            n = tripcount_stack[__idx_tripcount_stack_2_4]
            for i_seq_j = n:-1:1
                __idx_left_stack_2_0 = ((__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                left = left_stack[__idx_left_stack_2_0]
                __idx_right_stack_2_2 = ((__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                right = right_stack[__idx_right_stack_2_2]
                __idx_branch_stack_2_4 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + (div(nu2 - 1, 1) + 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                __branch_pre_4 = branch_stack[__idx_branch_stack_2_4]
                right = 0.0
                if __branch_pre_4 == 1
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    right = 0.0
                end
                __idx_branch_stack_2_8 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                __branch_pre_2 = branch_stack[__idx_branch_stack_2_8]
                left = 0.0
                if __branch_pre_2 == 1
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    left = 0.0
                end
                __idx_u_stack_2_0 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((div(val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                u[i_seq_j, i_seq_level] = u_stack[__idx_u_stack_2_0]
                hl2b = hl2b + f[i_seq_j, i_seq_level] * (0.5 * ub[i_seq_j, i_seq_level])
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * (0.5 * ub[i_seq_j, i_seq_level])
                leftb = leftb + 0.5 * ub[i_seq_j, i_seq_level]
                rightb = rightb + 0.5 * ub[i_seq_j, i_seq_level]
                ub[i_seq_j, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                    rightb = 0.0
                end
                __idx_right_stack_2_0 = (__tot_right_stack_1 + prefix_right_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                right = right_stack[__idx_right_stack_2_0]
                rightb = 0.0
                if __branch_pre_2 == 1
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftb = 0.0
                end
                __idx_left_stack_2_0 = (__tot_left_stack_1 + prefix_left_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (((i_seq_k - 1) * (div(val_n_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                left = left_stack[__idx_left_stack_2_0]
                leftb = 0.0
            end
        end
        __idx_tripcount_stack_2_9 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + 1) + 1
        nc = tripcount_stack[__idx_tripcount_stack_2_9]
        for j = nc + 1:-1:1
            __idx_cl_stack_2_0 = (prefix_cl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + ((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)
            cl = cl_stack[__idx_cl_stack_2_0]
            __idx_cr_stack_2_2 = (prefix_cr_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + ((div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1))) + ((j - 1) + 1)
            cr = cr_stack[__idx_cr_stack_2_2]
            jf = j * 2 - 1
            __idx_branch_stack_2_5 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (div((val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
            __branch_pre_5 = branch_stack[__idx_branch_stack_2_5]
            cr = 0.0
            if __branch_pre_5 == 1
                cr = u[j, i_seq_level + 1]
            else
                cr = 0.0
            end
            __idx_branch_stack_2_9 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            __branch_pre_3 = branch_stack[__idx_branch_stack_2_9]
            cl = 0.0
            if __branch_pre_3 == 1
                cl = u[j - 1, i_seq_level + 1]
            else
                cl = 0.0
            end
            __idx_u_stack_2_0 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + (div(val_nc_2[div(i_seq_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            u[jf, i_seq_level] = u_stack[__idx_u_stack_2_0]
            clb = clb + 0.5 * ub[jf, i_seq_level]
            crb = crb + 0.5 * ub[jf, i_seq_level]
            if __branch_pre_5 == 1
                ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + crb
                crb = 0.0
            end
            __idx_cr_stack_2_0 = prefix_cr_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + ((j - 1) + 1)
            cr = cr_stack[__idx_cr_stack_2_0]
            crb = 0.0
            if __branch_pre_3 == 1
                ub[j - 1, i_seq_level + 1] = ub[j - 1, i_seq_level + 1] + clb
                clb = 0.0
            end
            __idx_cl_stack_2_0 = prefix_cl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1] + ((j - 1) + 1)
            cl = cl_stack[__idx_cl_stack_2_0]
            clb = 0.0
        end
        __idx_tripcount_stack_2_12 = (__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + 1
        nc = tripcount_stack[__idx_tripcount_stack_2_12]
        for j = nc:-1:1
            jf = j * 2
            __idx_u_stack_2_0 = ((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            u[jf, i_seq_level] = u_stack[__idx_u_stack_2_0]
            ub[j, i_seq_level + 1] = ub[j, i_seq_level + 1] + ub[jf, i_seq_level]
        end
        __idx_hl2_stack_2_0 = ((__tot_hl2_stack_1 + 1) + prefix_hl2_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + 1
        hl2 = hl2_stack[__idx_hl2_stack_2_0]
        hlb = hlb + hl * hl2b
        hlb = hlb + hl * hl2b
        hl2b = 0.0
        __idx_hl_stack_2_0 = (__tot_hl_stack_1 + prefix_hl_stack_2[div(i_seq_level - (num_levels - 1), -1) + 1]) + 1
        hl = hl_stack[__idx_hl_stack_2_0]
        hlb = 0.5hlb
    end
    u[1, num_levels] = u_stack[__tot_u_stack_1 + 1]
    hl2b = hl2b + (0.5 * f[1, num_levels]) * ub[1, num_levels]
    fb[1, num_levels] = fb[1, num_levels] + (0.5hl2) * ub[1, num_levels]
    ub[1, num_levels] = 0.0
    hl2 = hl2_stack[__tot_hl2_stack_1 + 1]
    hlb = hlb + hl * hl2b
    hlb = hlb + hl * hl2b
    hl2b = 0.0
    for i_seq_level = num_levels - 1:-1:1
        __idx_left_stack_1_0 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + (((((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + 1
        left = left_stack[__idx_left_stack_1_0]
        __idx_right_stack_1_2 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + (((((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + 1
        right = right_stack[__idx_right_stack_1_2]
        __idx_hl_stack_1_0 = prefix_hl_stack_1[(i_seq_level - 1) + 1] + 1
        hl = hl_stack[__idx_hl_stack_1_0]
        hlb = 2.0hlb
        __idx_tripcount_stack_1_7 = (prefix_tripcount_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) + 1) + 1)) + 1
        nc = tripcount_stack[__idx_tripcount_stack_1_7]
        for j = nc:-1:1
            __idx_u_stack_1_0 = (prefix_u_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            u[j, i_seq_level + 1] = u_stack[__idx_u_stack_1_0]
            ub[j, i_seq_level + 1] = 0.0
        end
        __idx_tripcount_stack_1_10 = (prefix_tripcount_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) + 1)) + 1
        nc = tripcount_stack[__idx_tripcount_stack_1_10]
        for j = nc:-1:1
            jf = j * 2
            __idx_f_stack_1_0 = prefix_f_stack_1[(i_seq_level - 1) + 1] + ((j - 1) + 1)
            f[j, i_seq_level + 1] = f_stack[__idx_f_stack_1_0]
            rb[jf - 1, i_seq_level] = rb[jf - 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            rb[jf, i_seq_level] = rb[jf, i_seq_level] + 0.5 * fb[j, i_seq_level + 1]
            rb[jf + 1, i_seq_level] = rb[jf + 1, i_seq_level] + 0.25 * fb[j, i_seq_level + 1]
            fb[j, i_seq_level + 1] = 0.0
        end
        __idx_tripcount_stack_1_13 = (prefix_tripcount_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1)) + 1
        n = tripcount_stack[__idx_tripcount_stack_1_13]
        for j = n:-1:1
            __idx_left_stack_1_0 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + ((((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
            left = left_stack[__idx_left_stack_1_0]
            __idx_right_stack_1_2 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + ((((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
            right = right_stack[__idx_right_stack_1_2]
            __idx_branch_stack_1_4 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
            __branch_pre_4 = branch_stack[__idx_branch_stack_1_4]
            right = 0.0
            if __branch_pre_4 == 1
                right = u[j + 1, i_seq_level]
            else
                right = 0.0
            end
            __idx_branch_stack_1_8 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((j - 1) + 1)
            __branch_pre_2 = branch_stack[__idx_branch_stack_1_8]
            left = 0.0
            if __branch_pre_2 == 1
                left = u[j - 1, i_seq_level]
            else
                left = 0.0
            end
            __idx_r_stack_1_0 = prefix_r_stack_1[(i_seq_level - 1) + 1] + ((j - 1) + 1)
            r[j, i_seq_level] = r_stack[__idx_r_stack_1_0]
            fb[j, i_seq_level] = fb[j, i_seq_level] + rb[j, i_seq_level]
            ub[j, i_seq_level] = ub[j, i_seq_level] + 2.0 * ((1.0 / hl2) * -(rb[j, i_seq_level]))
            leftb = leftb + -((1.0 / hl2) * -(rb[j, i_seq_level]))
            rightb = rightb + -((1.0 / hl2) * -(rb[j, i_seq_level]))
            hl2b = hl2b + -(((2.0 * u[j, i_seq_level] - left) - right) / hl2 ^ 2) * -(rb[j, i_seq_level])
            rb[j, i_seq_level] = 0.0
            if __branch_pre_4 == 1
                ub[j + 1, i_seq_level] = ub[j + 1, i_seq_level] + rightb
                rightb = 0.0
            end
            __idx_right_stack_1_0 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + ((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1))) + ((j - 1) + 1)
            right = right_stack[__idx_right_stack_1_0]
            rightb = 0.0
            if __branch_pre_2 == 1
                ub[j - 1, i_seq_level] = ub[j - 1, i_seq_level] + leftb
                leftb = 0.0
            end
            __idx_left_stack_1_0 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + ((((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1))) + ((j - 1) + 1)
            left = left_stack[__idx_left_stack_1_0]
            leftb = 0.0
        end
        for i_seq_k = nu1:-1:1
            __idx_left_stack_1_0 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)
            left = left_stack[__idx_left_stack_1_0]
            __idx_right_stack_1_2 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + (((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + ((i_seq_k - 1) + 1)
            right = right_stack[__idx_right_stack_1_2]
            __idx_tripcount_stack_1_4 = prefix_tripcount_stack_1[(i_seq_level - 1) + 1] + ((i_seq_k - 1) + 1)
            n = tripcount_stack[__idx_tripcount_stack_1_4]
            for i_seq_j = n:-1:1
                __idx_left_stack_1_0 = (prefix_left_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                left = left_stack[__idx_left_stack_1_0]
                __idx_right_stack_1_2 = (prefix_right_stack_1[(i_seq_level - 1) + 1] + ((div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1))) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                right = right_stack[__idx_right_stack_1_2]
                __idx_branch_stack_1_4 = (prefix_branch_stack_1[(i_seq_level - 1) + 1] + (div(nu1 - 1, 1) + 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1)) + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                __branch_pre_4 = branch_stack[__idx_branch_stack_1_4]
                right = 0.0
                if __branch_pre_4 == 1
                    right = u[i_seq_j + 1, i_seq_level]
                else
                    right = 0.0
                end
                __idx_branch_stack_1_8 = prefix_branch_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                __branch_pre_2 = branch_stack[__idx_branch_stack_1_8]
                left = 0.0
                if __branch_pre_2 == 1
                    left = u[i_seq_j - 1, i_seq_level]
                else
                    left = 0.0
                end
                __idx_u_stack_1_0 = prefix_u_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                u[i_seq_j, i_seq_level] = u_stack[__idx_u_stack_1_0]
                hl2b = hl2b + f[i_seq_j, i_seq_level] * (0.5 * ub[i_seq_j, i_seq_level])
                fb[i_seq_j, i_seq_level] = fb[i_seq_j, i_seq_level] + hl2 * (0.5 * ub[i_seq_j, i_seq_level])
                leftb = leftb + 0.5 * ub[i_seq_j, i_seq_level]
                rightb = rightb + 0.5 * ub[i_seq_j, i_seq_level]
                ub[i_seq_j, i_seq_level] = 0.0
                if __branch_pre_4 == 1
                    ub[i_seq_j + 1, i_seq_level] = ub[i_seq_j + 1, i_seq_level] + rightb
                    rightb = 0.0
                end
                __idx_right_stack_1_0 = prefix_right_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                right = right_stack[__idx_right_stack_1_0]
                rightb = 0.0
                if __branch_pre_2 == 1
                    ub[i_seq_j - 1, i_seq_level] = ub[i_seq_j - 1, i_seq_level] + leftb
                    leftb = 0.0
                end
                __idx_left_stack_1_0 = prefix_left_stack_1[(i_seq_level - 1) + 1] + (((i_seq_k - 1) * (div(val_n_1[(i_seq_level - 1) + 1] - 1, 1) + 1) + (i_seq_j - 1)) + 1)
                left = left_stack[__idx_left_stack_1_0]
                leftb = 0.0
            end
        end
        __idx_hl2_stack_1_0 = prefix_hl2_stack_1[(i_seq_level - 1) + 1] + 1
        hl2 = hl2_stack[__idx_hl2_stack_1_0]
        hlb = hlb + hl * hl2b
        hlb = hlb + hl * hl2b
        hl2b = 0.0
    end
    h1b = h1b + hlb
    hlb = 0.0
    return h1b
end

function mg_vcycle(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    n = n * 2
    nl = nfine
    hl = h1
    for i_seq_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_seq_k = 1:nu1
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_seq_level]
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_seq_level]
            end
            r[j, i_seq_level] = f[j, i_seq_level] - ((2.0 * u[j, i_seq_level] - left) - right) / hl2
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        for j = 1:nc
            jf = j * 2
            f[j, i_seq_level + 1] = 0.25 * r[jf - 1, i_seq_level] + 0.5 * r[jf, i_seq_level] + 0.25 * r[jf + 1, i_seq_level]
        end
        for j = 1:nc
            u[j, i_seq_level + 1] = 0.0
        end
        nl = ncg
        hl = hl * 2.0
    end
    hl2 = hl * hl
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_seq_level = num_levels - 1:-1:1
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        for j = 1:nc
            jf = j * 2
            u[jf, i_seq_level] = u[jf, i_seq_level] + u[j, i_seq_level + 1]
        end
        for j = 1:nc + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_seq_level + 1]
            end
            cr = 0.0
            if j <= nc
                cr = u[j, i_seq_level + 1]
            end
            u[jf, i_seq_level] = u[jf, i_seq_level] + 0.5 * (cl + cr)
        end
        for i_seq_k = 1:nu2
            for i_seq_j = 1:n
                left = 0.0
                if i_seq_j > 1
                    left = u[i_seq_j - 1, i_seq_level]
                end
                right = 0.0
                if i_seq_j < n
                    right = u[i_seq_j + 1, i_seq_level]
                end
                u[i_seq_j, i_seq_level] = 0.5 * (hl2 * f[i_seq_j, i_seq_level] + left + right)
            end
        end
    end
end
