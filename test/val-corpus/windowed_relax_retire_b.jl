function initstacks_windowed_relax_retire_b(dx, num_passes, w0)
    w = w0
    dx2 = dx * dx
    prefix_branch_stack_1 = Vector{Int}(undef, div(num_passes - 1, 1) + 1)
    __tot_branch_stack_1 = 0
    prefix_left_stack_1 = Vector{Int}(undef, div(num_passes - 1, 1) + 1)
    __tot_left_stack_1 = 0
    prefix_right_stack_1 = Vector{Int}(undef, div(num_passes - 1, 1) + 1)
    __tot_right_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, div(num_passes - 1, 1) + 1)
    __tot_tripcount_stack_1 = 0
    val_w_1 = Vector{Int64}(undef, div(num_passes - 1, 1) + 1)
    for i_pass = 1:num_passes
        prefix_branch_stack_1[(i_pass - 1) + 1] = __tot_branch_stack_1
        prefix_left_stack_1[(i_pass - 1) + 1] = __tot_left_stack_1
        prefix_right_stack_1[(i_pass - 1) + 1] = __tot_right_stack_1
        prefix_tripcount_stack_1[(i_pass - 1) + 1] = __tot_tripcount_stack_1
        if mod(i_pass, 2) == 0
            w = w - 1
        end
        val_w_1[(i_pass - 1) + 1] = w
        __tot_branch_stack_1 = __tot_branch_stack_1 + ((1 + (div(w - 1, 1) + 1)) + (div(w - 1, 1) + 1))
        __tot_left_stack_1 = __tot_left_stack_1 + ((div(w - 1, 1) + 1) + 1)
        __tot_right_stack_1 = __tot_right_stack_1 + ((div(w - 1, 1) + 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + 1
    end
    branch_stack = Vector{Int64}(undef, __tot_branch_stack_1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1)
    left_stack = Vector{Float64}(undef, __tot_left_stack_1 + 1)
    right_stack = Vector{Float64}(undef, __tot_right_stack_1 + 1)
    return (branch_stack, tripcount_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_w_1)
end

function windowed_relax_retire_b(u, ub, f, fb, w0, num_passes, dx, dxb, n, nb, branch_stack, tripcount_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, __tot_branch_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, val_w_1)
    dx2 = 0.0
    left = 0.0
    right = 0.0
    dx2b = 0.0
    leftb = 0.0
    rightb = 0.0
    w = w0
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
            left = 0.0
            if i_j > 1
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_pass - 1) + 1] + 1) + ((i_j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                left = u[i_j - 1]
            else
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_pass - 1) + 1] + 1) + ((i_j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            right = 0.0
            if i_j < n
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_pass - 1) + 1] + (1 + (div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1))) + ((i_j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                right = u[i_j + 1]
            else
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_pass - 1) + 1] + (1 + (div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1))) + ((i_j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            u[i_j] = 0.5 * (dx2 * f[i_j] + left + right)
            __idx_left_stack_1_5 = prefix_left_stack_1[(i_pass - 1) + 1] + ((i_j - 1) + 1)
            left_stack[__idx_left_stack_1_5] = left
            __idx_right_stack_1_7 = prefix_right_stack_1[(i_pass - 1) + 1] + ((i_j - 1) + 1)
            right_stack[__idx_right_stack_1_7] = right
        end
        __idx_left_stack_1_4 = (prefix_left_stack_1[(i_pass - 1) + 1] + (div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1)) + 1
        left_stack[__idx_left_stack_1_4] = left
        __idx_right_stack_1_6 = (prefix_right_stack_1[(i_pass - 1) + 1] + (div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1)) + 1
        right_stack[__idx_right_stack_1_6] = right
    end
    left_stack[__tot_left_stack_1 + 1] = left
    right_stack[__tot_right_stack_1 + 1] = right
    left = left_stack[__tot_left_stack_1 + 1]
    right = right_stack[__tot_right_stack_1 + 1]
    for i_pass = num_passes:-1:1
        __idx_left_stack_1_0 = (prefix_left_stack_1[(i_pass - 1) + 1] + (div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1)) + 1
        left = left_stack[__idx_left_stack_1_0]
        __idx_right_stack_1_2 = (prefix_right_stack_1[(i_pass - 1) + 1] + (div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1)) + 1
        right = right_stack[__idx_right_stack_1_2]
        __idx_tripcount_stack_1_4 = prefix_tripcount_stack_1[(i_pass - 1) + 1] + 1
        w = tripcount_stack[__idx_tripcount_stack_1_4]
        for i_j = w:-1:1
            __idx_left_stack_1_0 = prefix_left_stack_1[(i_pass - 1) + 1] + ((i_j - 1) + 1)
            left = left_stack[__idx_left_stack_1_0]
            __idx_right_stack_1_2 = prefix_right_stack_1[(i_pass - 1) + 1] + ((i_j - 1) + 1)
            right = right_stack[__idx_right_stack_1_2]
            __idx_branch_stack_1_4 = (prefix_branch_stack_1[(i_pass - 1) + 1] + (1 + (div(val_w_1[(i_pass - 1) + 1] - 1, 1) + 1))) + ((i_j - 1) + 1)
            __branch_pre_4 = branch_stack[__idx_branch_stack_1_4]
            right = 0.0
            if __branch_pre_4 == 1
                right = u[i_j + 1]
            else
                right = 0.0
            end
            __idx_branch_stack_1_8 = (prefix_branch_stack_1[(i_pass - 1) + 1] + 1) + ((i_j - 1) + 1)
            __branch_pre_2 = branch_stack[__idx_branch_stack_1_8]
            left = 0.0
            if __branch_pre_2 == 1
                left = u[i_j - 1]
            else
                left = 0.0
            end
            dx2b = dx2b + f[i_j] * (0.5 * ub[i_j])
            fb[i_j] = fb[i_j] + dx2 * (0.5 * ub[i_j])
            leftb = leftb + 0.5 * ub[i_j]
            rightb = rightb + 0.5 * ub[i_j]
            ub[i_j] = 0.0
            if __branch_pre_4 == 1
                ub[i_j + 1] = ub[i_j + 1] + rightb
                rightb = 0.0
            end
            rightb = 0.0
            if __branch_pre_2 == 1
                ub[i_j - 1] = ub[i_j - 1] + leftb
                leftb = 0.0
            end
            leftb = 0.0
        end
        __idx_branch_stack_1_7 = prefix_branch_stack_1[(i_pass - 1) + 1] + 1
        __branch = branch_stack[__idx_branch_stack_1_7]
        if __branch == 1
        end
    end
    dxb = dxb + dx * dx2b
    dxb = dxb + dx * dx2b
    dx2b = 0.0
    return (dxb, nb)
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
