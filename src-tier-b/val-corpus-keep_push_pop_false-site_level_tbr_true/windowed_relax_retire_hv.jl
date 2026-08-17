function initstacks_windowed_relax_retire_b(num_passes)
    branch_stack = Vector{Int64}()
    tripcount_stack = Vector{Int64}(undef, div(num_passes - 1, 1) + 1)
    left_stack = Vector{Float64}()
    right_stack = Vector{Float64}()
    return (branch_stack, tripcount_stack, left_stack, right_stack)
end

function windowed_relax_retire_hv(u, ub, f, fb, w0, num_passes, dx, dxb, n, nb, ud, ubd, fd, fbd, dxd, dxbd, nd, nbd, branch_stack, tripcount_stack, left_stack, right_stack)
    left_stack_d = Vector{Float64}()
    right_stack_d = Vector{Float64}()
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
    dx2d = dx * dxd + dx * dxd
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
            push!(left_stack_d, leftd)
            push!(left_stack, left)
            leftd = 0.0
            left = 0.0
            if i_seq_j > 1
                push!(branch_stack, 1)
                push!(left_stack_d, leftd)
                push!(left_stack, left)
                leftd = ud[i_seq_j - 1]
                left = u[i_seq_j - 1]
            else
                push!(branch_stack, 0)
            end
            push!(right_stack_d, rightd)
            push!(right_stack, right)
            rightd = 0.0
            right = 0.0
            if i_seq_j < n
                push!(branch_stack, 1)
                push!(right_stack_d, rightd)
                push!(right_stack, right)
                rightd = ud[i_seq_j + 1]
                right = u[i_seq_j + 1]
            else
                push!(branch_stack, 0)
            end
            ud[i_seq_j] = 0.5 * (((f[i_seq_j] * dx2d + dx2 * fd[i_seq_j]) + leftd) + rightd)
            u[i_seq_j] = 0.5 * (dx2 * f[i_seq_j] + left + right)
            push!(left_stack_d, leftd)
            push!(left_stack, left)
            push!(right_stack_d, rightd)
            push!(right_stack, right)
        end
        push!(left_stack_d, leftd)
        push!(left_stack, left)
        push!(right_stack_d, rightd)
        push!(right_stack, right)
    end
    push!(left_stack_d, leftd)
    push!(left_stack, left)
    push!(right_stack_d, rightd)
    push!(right_stack, right)
    leftd = pop!(left_stack_d)
    left = pop!(left_stack)
    rightd = pop!(right_stack_d)
    right = pop!(right_stack)
    for i_seq_pass = num_passes:-1:1
        leftd = pop!(left_stack_d)
        left = pop!(left_stack)
        rightd = pop!(right_stack_d)
        right = pop!(right_stack)
        w = tripcount_stack[(i_seq_pass - 1) + 1]
        for i_seq_j = w:-1:1
            leftd = pop!(left_stack_d)
            left = pop!(left_stack)
            rightd = pop!(right_stack_d)
            right = pop!(right_stack)
            __branch_pre_4 = pop!(branch_stack)
            rightd = 0.0
            right = 0.0
            if __branch_pre_4 == 1
                rightd = ud[i_seq_j + 1]
                right = u[i_seq_j + 1]
            else
                rightd = 0.0
                right = 0.0
            end
            __branch_pre_2 = pop!(branch_stack)
            leftd = 0.0
            left = 0.0
            if __branch_pre_2 == 1
                leftd = ud[i_seq_j - 1]
                left = u[i_seq_j - 1]
            else
                leftd = 0.0
                left = 0.0
            end
            dx2bd = dx2bd + ((0.5 * ub[i_seq_j]) * fd[i_seq_j] + f[i_seq_j] * (0.5 * ubd[i_seq_j]))
            dx2b = dx2b + f[i_seq_j] * (0.5 * ub[i_seq_j])
            fbd[i_seq_j] = fbd[i_seq_j] + ((0.5 * ub[i_seq_j]) * dx2d + dx2 * (0.5 * ubd[i_seq_j]))
            fb[i_seq_j] = fb[i_seq_j] + dx2 * (0.5 * ub[i_seq_j])
            leftbd = leftbd + 0.5 * ubd[i_seq_j]
            leftb = leftb + 0.5 * ub[i_seq_j]
            rightbd = rightbd + 0.5 * ubd[i_seq_j]
            rightb = rightb + 0.5 * ub[i_seq_j]
            ubd[i_seq_j] = 0.0
            ub[i_seq_j] = 0.0
            if __branch_pre_4 == 1
                ubd[i_seq_j + 1] = ubd[i_seq_j + 1] + rightbd
                ub[i_seq_j + 1] = ub[i_seq_j + 1] + rightb
                rightbd = 0.0
                rightb = 0.0
            end
            rightd = pop!(right_stack_d)
            right = pop!(right_stack)
            rightbd = 0.0
            rightb = 0.0
            if __branch_pre_2 == 1
                ubd[i_seq_j - 1] = ubd[i_seq_j - 1] + leftbd
                ub[i_seq_j - 1] = ub[i_seq_j - 1] + leftb
                leftbd = 0.0
                leftb = 0.0
            end
            leftd = pop!(left_stack_d)
            left = pop!(left_stack)
            leftbd = 0.0
            leftb = 0.0
        end
        __branch = pop!(branch_stack)
        if __branch == 1
        end
    end
    dxbd = dxbd + (dx2b * dxd + dx * dx2bd)
    dxb = dxb + dx * dx2b
    dxbd = dxbd + (dx2b * dxd + dx * dx2bd)
    dxb = dxb + dx * dx2b
    dx2bd = 0.0
    dx2b = 0.0
    return (dxb, dxbd, nb, nbd)
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
