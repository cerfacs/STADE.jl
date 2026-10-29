function initstacks_if_scalar_then_loop_if_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(2 - 1, 1) + 1))
    left_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1))
    return (branch_stack, left_stack)
end

function if_scalar_then_loop_if_b(u, ub, x, xb, y, yb, rhs, rhsb, i_n, branch_stack, left_stack)
    left = 0.0
    leftb = 0.0
    for i = 1:i_n
        left = 0.0
        if i > 1
            __idx_branch_stack_0 = (i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            left = u[i - 1]
        else
            __idx_branch_stack_0 = (i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
        for j = 1:2
            if j > 1
                __icse_0 = div(i_n - 1, 1) + 1
                __idx_branch_stack_0 = max(0, __icse_0) + (((i - 1) + (j - 1) * __icse_0) + 1)
                branch_stack[__idx_branch_stack_0] = 1
                __cse_1 = x[j]
                y[i] = y[i] + __cse_1 * __cse_1
            else
                __icse_2 = div(i_n - 1, 1) + 1
                __idx_branch_stack_0 = max(0, __icse_2) + (((i - 1) + (j - 1) * __icse_2) + 1)
                branch_stack[__idx_branch_stack_0] = 0
            end
        end
        u[i] = u[i] + rhs[i] * left * left
        __idx_left_stack_4 = (i - 1) + 1
        left_stack[__idx_left_stack_4] = left
    end
    for i = i_n:-1:1
        __idx_left_stack_0 = (i - 1) + 1
        left = left_stack[__idx_left_stack_0]
        __cse_3 = ub[i]
        rhsb[i] = rhsb[i] + (left * left) * __cse_3
        __cse_4 = (rhs[i] * left) * __cse_3
        leftb = leftb + __cse_4
        leftb = leftb + __cse_4
        for j = 2:-1:1
            __icse_5 = div(i_n - 1, 1) + 1
            __idx_branch_stack_0 = max(0, __icse_5) + (((i - 1) + (j - 1) * __icse_5) + 1)
            __branch = branch_stack[__idx_branch_stack_0]
            if __branch == 1
                __cse_6 = x[j] * yb[i]
                xb[j] = xb[j] + __cse_6
                xb[j] = xb[j] + __cse_6
            end
        end
        __idx_branch_stack_6 = (i - 1) + 1
        __branch = branch_stack[__idx_branch_stack_6]
        if __branch == 1
            __oldb_0 = leftb
            leftb = 0.0
            ub[i - 1] = ub[i - 1] + __oldb_0
        end
        leftb = 0.0
    end
    return nothing
end

function if_scalar_then_loop_if(u, x, y, rhs, i_n)
    for i = 1:i_n
        left = 0.0
        if i > 1
            left = u[i - 1]
        end
        for j = 1:2
            if j > 1
                y[i] = y[i] + x[j] * x[j]
            end
        end
        u[i] = u[i] + rhs[i] * left * left
    end
end
