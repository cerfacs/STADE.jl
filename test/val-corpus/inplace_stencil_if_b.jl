function initstacks_inplace_stencil_if_b(i_n, i_nu)
    branch_stack = Vector{Int64}(undef, max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1) + max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    left_stack = Vector{Float64}(undef, max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    right_stack = Vector{Float64}(undef, max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    return (branch_stack, left_stack, right_stack)
end

function inplace_stencil_if_b(u, ub, rhs, rhsb, i_nu, i_n, branch_stack, left_stack, right_stack)
    left = 0.0
    right = 0.0
    leftb = 0.0
    rightb = 0.0
    for i_k = 1:i_nu
        for i_j = 1:i_n
            left = 0.0
            if i_j > 1
                __idx_branch_stack_0 = ((i_k - 1) + (i_j - 1) * (div(i_nu - 1, 1) + 1)) + 1
                branch_stack[__idx_branch_stack_0] = 1
                left = u[i_j - 1]
            else
                __idx_branch_stack_0 = ((i_k - 1) + (i_j - 1) * (div(i_nu - 1, 1) + 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            right = 0.0
            if i_j < i_n
                __icse_0 = div(i_nu - 1, 1) + 1
                __idx_branch_stack_0 = max(0, __icse_0) * max(0, div(i_n - 1, 1) + 1) + (((i_k - 1) + (i_j - 1) * __icse_0) + 1)
                branch_stack[__idx_branch_stack_0] = 1
                right = u[i_j + 1]
            else
                __icse_1 = div(i_nu - 1, 1) + 1
                __idx_branch_stack_0 = max(0, __icse_1) * max(0, div(i_n - 1, 1) + 1) + (((i_k - 1) + (i_j - 1) * __icse_1) + 1)
                branch_stack[__idx_branch_stack_0] = 0
            end
            u[i_j] = 0.5 * (rhs[i_j] + left * right + left)
            __icse_2 = ((i_k - 1) + (i_j - 1) * (div(i_nu - 1, 1) + 1)) + 1
            __idx_left_stack_5 = __icse_2
            left_stack[__idx_left_stack_5] = left
            __idx_right_stack_7 = __icse_2
            right_stack[__idx_right_stack_7] = right
        end
    end
    for i_k = i_nu:-1:1
        for i_j = i_n:-1:1
            __icse_3 = div(i_nu - 1, 1) + 1
            __icse_4 = ((i_k - 1) + (i_j - 1) * __icse_3) + 1
            __idx_left_stack_0 = __icse_4
            left = left_stack[__idx_left_stack_0]
            __idx_right_stack_2 = __icse_4
            right = right_stack[__idx_right_stack_2]
            __oldb_0 = ub[i_j]
            ub[i_j] = 0.0
            __cse_5 = 0.5__oldb_0
            rhsb[i_j] = rhsb[i_j] + __cse_5
            leftb = leftb + right * __cse_5
            rightb = rightb + left * __cse_5
            leftb = leftb + __cse_5
            __idx_branch_stack_10 = max(0, __icse_3) * max(0, div(i_n - 1, 1) + 1) + __icse_4
            __branch = branch_stack[__idx_branch_stack_10]
            if __branch == 1
                __oldb_0 = rightb
                rightb = 0.0
                ub[i_j + 1] = ub[i_j + 1] + __oldb_0
            end
            rightb = 0.0
            __idx_branch_stack_14 = ((i_k - 1) + (i_j - 1) * (div(i_nu - 1, 1) + 1)) + 1
            __branch = branch_stack[__idx_branch_stack_14]
            if __branch == 1
                __oldb_0 = leftb
                leftb = 0.0
                ub[i_j - 1] = ub[i_j - 1] + __oldb_0
            end
            leftb = 0.0
        end
    end
    return nothing
end

function inplace_stencil_if(u, rhs, i_nu, i_n)
    for i_k = 1:i_nu
        for i_j = 1:i_n
            left = 0.0
            if i_j > 1
                left = u[i_j - 1]
            end
            right = 0.0
            if i_j < i_n
                right = u[i_j + 1]
            end
            u[i_j] = 0.5 * (rhs[i_j] + left * right + left)
        end
    end
end
