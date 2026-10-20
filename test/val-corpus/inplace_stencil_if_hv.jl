function initstacks_inplace_stencil_if_b(i_n, i_nu)
    branch_stack = Vector{Int64}(undef, max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1) + max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    left_stack = Vector{Float64}(undef, max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    right_stack = Vector{Float64}(undef, max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    return (branch_stack, left_stack, right_stack)
end

function inplace_stencil_if_hv(u, ub, rhs, rhsb, i_nu, i_n, ud, ubd, rhsd, rhsbd, branch_stack, left_stack, right_stack)
    left_stack_d = Vector{Float64}(undef, length(left_stack))
    right_stack_d = Vector{Float64}(undef, length(right_stack))
    left = 0.0
    right = 0.0
    leftb = 0.0
    rightb = 0.0
    leftd = 0.0
    leftbd = 0.0
    rightd = 0.0
    rightbd = 0.0
    for i_k = 1:i_nu
        for i_j = 1:i_n
            leftd = 0.0
            left = 0.0
            if i_j > 1
                __idx_branch_stack_0 = ((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_j - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 1
                leftd = ud[i_j - 1]
                left = u[i_j - 1]
            else
                __idx_branch_stack_0 = ((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_j - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            rightd = 0.0
            right = 0.0
            if i_j < i_n
                __icse_0 = div(i_n - 1, 1) + 1
                __idx_branch_stack_0 = max(0, div(i_nu - 1, 1) + 1) * max(0, __icse_0) + (((i_k - 1) * __icse_0 + (i_j - 1)) + 1)
                branch_stack[__idx_branch_stack_0] = 1
                rightd = ud[i_j + 1]
                right = u[i_j + 1]
            else
                __icse_1 = div(i_n - 1, 1) + 1
                __idx_branch_stack_0 = max(0, div(i_nu - 1, 1) + 1) * max(0, __icse_1) + (((i_k - 1) * __icse_1 + (i_j - 1)) + 1)
                branch_stack[__idx_branch_stack_0] = 0
            end
            ud[i_j] = 0.5 * ((rhsd[i_j] + (right * leftd + left * rightd)) + leftd)
            u[i_j] = 0.5 * (rhs[i_j] + left * right + left)
            __icse_2 = ((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_j - 1)) + 1
            __idx_left_stack_5 = __icse_2
            left_stack_d[__idx_left_stack_5] = leftd
            left_stack[__idx_left_stack_5] = left
            __idx_right_stack_7 = __icse_2
            right_stack_d[__idx_right_stack_7] = rightd
            right_stack[__idx_right_stack_7] = right
        end
    end
    for i_k = i_nu:-1:1
        for i_j = i_n:-1:1
            __icse_3 = div(i_n - 1, 1) + 1
            __icse_4 = ((i_k - 1) * __icse_3 + (i_j - 1)) + 1
            __idx_left_stack_0 = __icse_4
            leftd = left_stack_d[__idx_left_stack_0]
            left = left_stack[__idx_left_stack_0]
            __idx_right_stack_2 = __icse_4
            rightd = right_stack_d[__idx_right_stack_2]
            right = right_stack[__idx_right_stack_2]
            __oldb_0d = ubd[i_j]
            __oldb_0 = ub[i_j]
            ubd[i_j] = 0.0
            ub[i_j] = 0.0
            __cse_5d = 0.5__oldb_0d
            __cse_5 = 0.5__oldb_0
            rhsbd[i_j] = rhsbd[i_j] + __cse_5d
            rhsb[i_j] = rhsb[i_j] + __cse_5
            leftbd = leftbd + (__cse_5 * rightd + right * __cse_5d)
            leftb = leftb + right * __cse_5
            rightbd = rightbd + (__cse_5 * leftd + left * __cse_5d)
            rightb = rightb + left * __cse_5
            leftbd = leftbd + __cse_5d
            leftb = leftb + __cse_5
            __idx_branch_stack_10 = max(0, div(i_nu - 1, 1) + 1) * max(0, __icse_3) + __icse_4
            __branch = branch_stack[__idx_branch_stack_10]
            if __branch == 1
                __oldb_0d = rightbd
                __oldb_0 = rightb
                rightbd = 0.0
                rightb = 0.0
                ubd[i_j + 1] = ubd[i_j + 1] + __oldb_0d
                ub[i_j + 1] = ub[i_j + 1] + __oldb_0
            end
            rightbd = 0.0
            rightb = 0.0
            __idx_branch_stack_14 = ((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_j - 1)) + 1
            __branch = branch_stack[__idx_branch_stack_14]
            if __branch == 1
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
