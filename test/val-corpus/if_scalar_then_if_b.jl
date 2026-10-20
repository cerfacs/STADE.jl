function initstacks_if_scalar_then_if_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    left_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1))
    return (branch_stack, left_stack)
end

function if_scalar_then_if_b(u, ub, x, xb, y, yb, rhs, rhsb, i_n, branch_stack, left_stack)
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
        if i > 2
            __idx_branch_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((i - 1) + 1)
            branch_stack[__idx_branch_stack_0] = 1
            __cse_0 = x[i]
            y[i] = y[i] + __cse_0 * __cse_0
        else
            __idx_branch_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((i - 1) + 1)
            branch_stack[__idx_branch_stack_0] = 0
        end
        u[i] = u[i] + rhs[i] * left * left
        __idx_left_stack_4 = (i - 1) + 1
        left_stack[__idx_left_stack_4] = left
    end
    for i = i_n:-1:1
        __icse_1 = (i - 1) + 1
        __idx_left_stack_0 = __icse_1
        left = left_stack[__idx_left_stack_0]
        __cse_2 = ub[i]
        rhsb[i] = rhsb[i] + (left * left) * __cse_2
        __cse_3 = (rhs[i] * left) * __cse_2
        leftb = leftb + __cse_3
        leftb = leftb + __cse_3
        __idx_branch_stack_5 = max(0, div(i_n - 1, 1) + 1) + __icse_1
        __branch = branch_stack[__idx_branch_stack_5]
        if __branch == 1
            __cse_4 = x[i] * yb[i]
            xb[i] = xb[i] + __cse_4
            xb[i] = xb[i] + __cse_4
        end
        __idx_branch_stack_8 = (i - 1) + 1
        __branch = branch_stack[__idx_branch_stack_8]
        if __branch == 1
            __oldb_0 = leftb
            leftb = 0.0
            ub[i - 1] = ub[i - 1] + __oldb_0
        end
        leftb = 0.0
    end
    return nothing
end

function if_scalar_then_if(u, x, y, rhs, i_n)
    for i = 1:i_n
        left = 0.0
        if i > 1
            left = u[i - 1]
        end
        if i > 2
            y[i] = y[i] + x[i] * x[i]
        end
        u[i] = u[i] + rhs[i] * left * left
    end
end
