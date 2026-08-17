function initstacks_windowed_relax_retire_b(num_passes)
    branch_stack = Vector{Int64}()
    tripcount_stack = Vector{Int64}(undef, div(num_passes - 1, 1) + 1)
    left_stack = Vector{Float64}()
    right_stack = Vector{Float64}()
    return (branch_stack, tripcount_stack, left_stack, right_stack)
end

function windowed_relax_retire_b(u, ub, f, fb, w0, num_passes, dx, dxb, n, nb, branch_stack, tripcount_stack, left_stack, right_stack)
    dx2 = 0.0
    left = 0.0
    right = 0.0
    dx2b = 0.0
    leftb = 0.0
    rightb = 0.0
    w = w0
    dx2 = dx * dx
    for i_seq_pass = 1:num_passes
        if mod(i_seq_pass, 2) == 0
            push!(branch_stack, 1)
            w = w - 1
        else
            push!(branch_stack, 0)
        end
        tripcount_stack[(i_seq_pass - 1) + 1] = w
        for i_seq_j = 1:w
            push!(left_stack, left)
            left = 0.0
            if i_seq_j > 1
                push!(branch_stack, 1)
                push!(left_stack, left)
                left = u[i_seq_j - 1]
            else
                push!(branch_stack, 0)
            end
            push!(right_stack, right)
            right = 0.0
            if i_seq_j < n
                push!(branch_stack, 1)
                push!(right_stack, right)
                right = u[i_seq_j + 1]
            else
                push!(branch_stack, 0)
            end
            u[i_seq_j] = 0.5 * (dx2 * f[i_seq_j] + left + right)
            push!(left_stack, left)
            push!(right_stack, right)
        end
        push!(left_stack, left)
        push!(right_stack, right)
    end
    push!(left_stack, left)
    push!(right_stack, right)
    left = pop!(left_stack)
    right = pop!(right_stack)
    for i_seq_pass = num_passes:-1:1
        left = pop!(left_stack)
        right = pop!(right_stack)
        w = tripcount_stack[(i_seq_pass - 1) + 1]
        for i_seq_j = w:-1:1
            left = pop!(left_stack)
            right = pop!(right_stack)
            __branch_pre_4 = pop!(branch_stack)
            right = 0.0
            if __branch_pre_4 == 1
                right = u[i_seq_j + 1]
            else
                right = 0.0
            end
            __branch_pre_2 = pop!(branch_stack)
            left = 0.0
            if __branch_pre_2 == 1
                left = u[i_seq_j - 1]
            else
                left = 0.0
            end
            dx2b = dx2b + f[i_seq_j] * (0.5 * ub[i_seq_j])
            fb[i_seq_j] = fb[i_seq_j] + dx2 * (0.5 * ub[i_seq_j])
            leftb = leftb + 0.5 * ub[i_seq_j]
            rightb = rightb + 0.5 * ub[i_seq_j]
            ub[i_seq_j] = 0.0
            if __branch_pre_4 == 1
                ub[i_seq_j + 1] = ub[i_seq_j + 1] + rightb
                rightb = 0.0
            end
            right = pop!(right_stack)
            rightb = 0.0
            if __branch_pre_2 == 1
                ub[i_seq_j - 1] = ub[i_seq_j - 1] + leftb
                leftb = 0.0
            end
            left = pop!(left_stack)
            leftb = 0.0
        end
        __branch = pop!(branch_stack)
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
    for i_seq_pass = 1:num_passes
        if mod(i_seq_pass, 2) == 0
            w = w - 1
        end
        for i_seq_j = 1:w
            left = 0.0
            if i_seq_j > 1
                left = u[i_seq_j - 1]
            end
            right = 0.0
            if i_seq_j < n
                right = u[i_seq_j + 1]
            end
            u[i_seq_j] = 0.5 * (dx2 * f[i_seq_j] + left + right)
        end
    end
end
