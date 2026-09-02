function initstacks_windowed_relax_retire_b(dx, num_passes, w0)
    w = w0
    dx2 = dx * dx
    prefix_branch_stack_1 = Vector{Int}(undef, max(0, div(num_passes - 1, 1) + 1))
    __tot_branch_stack_1 = 0
    prefix_left_stack_1 = Vector{Int}(undef, max(0, div(num_passes - 1, 1) + 1))
    __tot_left_stack_1 = 0
    prefix_right_stack_1 = Vector{Int}(undef, max(0, div(num_passes - 1, 1) + 1))
    __tot_right_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(num_passes - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    val_w_1 = Vector{Int64}(undef, max(0, div(num_passes - 1, 1) + 1))
    for i_pass = 1:num_passes
        prefix_branch_stack_1[(i_pass - 1) + 1] = __tot_branch_stack_1
        prefix_left_stack_1[(i_pass - 1) + 1] = __tot_left_stack_1
        prefix_right_stack_1[(i_pass - 1) + 1] = __tot_right_stack_1
        prefix_tripcount_stack_1[(i_pass - 1) + 1] = __tot_tripcount_stack_1
        if mod(i_pass, 2) == 0
            w = w - 1
        end
        val_w_1[(i_pass - 1) + 1] = w
        __tot_branch_stack_1 = __tot_branch_stack_1 + ((1 + max(0, div(w - 1, 1) + 1)) + max(0, div(w - 1, 1) + 1))
        __tot_left_stack_1 = __tot_left_stack_1 + max(0, div(w - 1, 1) + 1)
        __tot_right_stack_1 = __tot_right_stack_1 + max(0, div(w - 1, 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    branch_stack = Vector{Int64}(undef, __tot_branch_stack_1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    left_stack = Vector{Float64}(undef, __tot_left_stack_1)
    right_stack = Vector{Float64}(undef, __tot_right_stack_1)
    return (branch_stack, tripcount_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_w_1)
end

function windowed_relax_retire_hv(u, ub, f, fb, w0, num_passes, dx, dxb, n, nb, ud, ubd, fd, fbd, dxd, dxbd, nd, nbd, branch_stack, tripcount_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_w_1)
    left_stack_d = Vector{Float64}(undef, length(left_stack))
    right_stack_d = Vector{Float64}(undef, length(right_stack))
    dx2 = 0.0
    left = 0.0
    right = 0.0
    dx2b = 0.0
    leftb = 0.0
    rightb = 0.0
    dx2d = 0.0
    dx2bd = 0.0
    leftd = 0.0
    leftbd = 0.0
    rightd = 0.0
    rightbd = 0.0
    w = w0
    __hcse_0 = dx * dxd
    dx2d = __hcse_0 + __hcse_0
    dx2 = dx * dx
    for i_pass = 1:num_passes
        if mod(i_pass, 2) == 0
            __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_pass - 1) + 1] + 1
            branch_stack[__idx_branch_stack_1_0] = 1
            w = w - 1
        else
            __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_pass - 1) + 1] + 1
            branch_stack[__idx_branch_stack_1_0] = 0
        end
        __idx_tripcount_stack_1_1 = prefix_tripcount_stack_1[(i_pass - 1) + 1] + 1
        tripcount_stack[__idx_tripcount_stack_1_1] = w
        for i_j = 1:w
            leftd = 0.0
            left = 0.0
            if i_j > 1
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_pass - 1) + 1] + 1) + ((i_j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                leftd = ud[i_j - 1]
                left = u[i_j - 1]
            else
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_pass - 1) + 1] + 1) + ((i_j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            rightd = 0.0
            right = 0.0
            if i_j < n
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_pass - 1) + 1] + (1 + max(0, div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1))) + ((i_j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                rightd = ud[i_j + 1]
                right = u[i_j + 1]
            else
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_pass - 1) + 1] + (1 + max(0, div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1))) + ((i_j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            __hcse_1 = f[i_j]
            ud[i_j] = 0.5 * (((__hcse_1 * dx2d + dx2 * fd[i_j]) + leftd) + rightd)
            u[i_j] = 0.5 * (dx2 * __hcse_1 + left + right)
            __icse_0 = (i_j - 1) + 1
            __idx_left_stack_1_5 = prefix_left_stack_1[(i_pass - 1) + 1] + __icse_0
            left_stack_d[__idx_left_stack_1_5] = leftd
            left_stack[__idx_left_stack_1_5] = left
            __idx_right_stack_1_7 = prefix_right_stack_1[(i_pass - 1) + 1] + __icse_0
            right_stack_d[__idx_right_stack_1_7] = rightd
            right_stack[__idx_right_stack_1_7] = right
        end
    end
    for i_pass = num_passes:-1:1
        __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_pass - 1) + 1] + 1
        w = tripcount_stack[__idx_tripcount_stack_1_0]
        for i_j = w:-1:1
            __icse_1 = (i_j - 1) + 1
            __idx_left_stack_1_0 = prefix_left_stack_1[(i_pass - 1) + 1] + __icse_1
            leftd = left_stack_d[__idx_left_stack_1_0]
            left = left_stack[__idx_left_stack_1_0]
            __idx_right_stack_1_2 = prefix_right_stack_1[(i_pass - 1) + 1] + __icse_1
            rightd = right_stack_d[__idx_right_stack_1_2]
            right = right_stack[__idx_right_stack_1_2]
            __idx_branch_stack_1_4 = (prefix_branch_stack_1[(i_pass - 1) + 1] + (1 + max(0, div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1))) + __icse_1
            __branch_pre_4 = branch_stack[__idx_branch_stack_1_4]
            rightd = 0.0
            right = 0.0
            if __branch_pre_4 == 1
                rightd = ud[i_j + 1]
                right = u[i_j + 1]
            else
                rightd = 0.0
                right = 0.0
            end
            __idx_branch_stack_1_8 = (prefix_branch_stack_1[(i_pass - 1) + 1] + 1) + ((i_j - 1) + 1)
            __branch_pre_2 = branch_stack[__idx_branch_stack_1_8]
            leftd = 0.0
            left = 0.0
            if __branch_pre_2 == 1
                leftd = ud[i_j - 1]
                left = u[i_j - 1]
            else
                leftd = 0.0
                left = 0.0
            end
            __oldb_0d = ubd[i_j]
            __oldb_0 = ub[i_j]
            ubd[i_j] = 0.0
            ub[i_j] = 0.0
            __cse_2d = 0.5__oldb_0d
            __cse_2 = 0.5__oldb_0
            __hcse_2 = f[i_j]
            dx2bd = dx2bd + (__cse_2 * fd[i_j] + __hcse_2 * __cse_2d)
            dx2b = dx2b + __hcse_2 * __cse_2
            fbd[i_j] = fbd[i_j] + (__cse_2 * dx2d + dx2 * __cse_2d)
            fb[i_j] = fb[i_j] + dx2 * __cse_2
            leftbd = leftbd + __cse_2d
            leftb = leftb + __cse_2
            rightbd = rightbd + __cse_2d
            rightb = rightb + __cse_2
            if __branch_pre_4 == 1
                __oldb_0d = rightbd
                __oldb_0 = rightb
                rightbd = 0.0
                rightb = 0.0
                ubd[i_j + 1] = ubd[i_j + 1] + __oldb_0d
                ub[i_j + 1] = ub[i_j + 1] + __oldb_0
            end
            rightbd = 0.0
            rightb = 0.0
            if __branch_pre_2 == 1
                __oldb_0d = leftbd
                __oldb_0 = leftb
                leftbd = 0.0
                leftb = 0.0
                ubd[i_j - 1] = ubd[i_j - 1] + __oldb_0d
                ub[i_j - 1] = ub[i_j - 1] + __oldb_0
            end
            leftbd = 0.0
            leftb = 0.0
        end
        __idx_branch_stack_1_3 = prefix_branch_stack_1[(i_pass - 1) + 1] + 1
        __branch = branch_stack[__idx_branch_stack_1_3]
        if __branch == 1
        end
    end
    __oldb_0d = dx2bd
    __oldb_0 = dx2b
    dx2bd = 0.0
    dx2b = 0.0
    __cse_3d = __oldb_0 * dxd + dx * __oldb_0d
    __cse_3 = dx * __oldb_0
    dxbd = dxbd + __cse_3d
    dxb = dxb + __cse_3
    dxbd = dxbd + __cse_3d
    dxb = dxb + __cse_3
    return (dxb, dxbd, nb, nbd)
end

function windowed_relax_retire(u, f, w0, num_passes, dx, n)
    w = w0
    dx2 = dx * dx
    for i_pass = 1:num_passes
        if mod(i_pass, 2) == 0
            w = w - 1
        end
        for i_j = 1:w
            left = 0.0
            if i_j > 1
                left = u[i_j - 1]
            end
            right = 0.0
            if i_j < n
                right = u[i_j + 1]
            end
            u[i_j] = 0.5 * (dx2 * f[i_j] + left + right)
        end
    end
end
