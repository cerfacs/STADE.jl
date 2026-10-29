function initstacks_nest_if_scalar_b(i_n, i_nu)
    branch_stack = Vector{Int64}(undef, ((max(0, div(i_nu - 1, 1) + 1) + max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    left_stack = Vector{Float64}(undef, max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    right_stack = Vector{Float64}(undef, max(0, div(i_nu - 1, 1) + 1) * max(0, div(i_n - 1, 1) + 1))
    return (branch_stack, left_stack, right_stack)
end

function nest_if_scalar_b(u, ub, rhs, rhsb, i_nu, i_n, branch_stack, left_stack, right_stack)
    left = 0.0
    right = 0.0
    leftb = 0.0
    rightb = 0.0
    for i_k = 1:i_nu
        if i_nu >= 0
            __idx_branch_stack_0 = (i_k - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            for i_j = 1:i_n
                if i_nu >= 0
                    __idx_branch_stack_0 = max(0, div(i_nu - 1, 1) + 1) + (((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_0] = 1
                    left = 0.0
                    if i_j > 1
                        __icse_0 = max(0, div(i_nu - 1, 1) + 1)
                        __icse_1 = div(i_n - 1, 1) + 1
                        __idx_branch_stack_0 = (__icse_0 + __icse_0 * max(0, __icse_1)) + (((i_k - 1) * __icse_1 + (i_j - 1)) + 1)
                        branch_stack[__idx_branch_stack_0] = 1
                        left = u[i_j - 1]
                    else
                        __icse_2 = max(0, div(i_nu - 1, 1) + 1)
                        __icse_3 = div(i_n - 1, 1) + 1
                        __idx_branch_stack_0 = (__icse_2 + __icse_2 * max(0, __icse_3)) + (((i_k - 1) * __icse_3 + (i_j - 1)) + 1)
                        branch_stack[__idx_branch_stack_0] = 0
                    end
                    right = 0.0
                    if i_j < i_n
                        __icse_4 = max(0, div(i_nu - 1, 1) + 1)
                        __icse_5 = div(i_n - 1, 1) + 1
                        __icse_6 = __icse_4 * max(0, __icse_5)
                        __idx_branch_stack_0 = ((__icse_4 + __icse_6) + __icse_6) + (((i_k - 1) * __icse_5 + (i_j - 1)) + 1)
                        branch_stack[__idx_branch_stack_0] = 1
                        right = u[i_j + 1]
                    else
                        __icse_7 = max(0, div(i_nu - 1, 1) + 1)
                        __icse_8 = div(i_n - 1, 1) + 1
                        __icse_9 = __icse_7 * max(0, __icse_8)
                        __idx_branch_stack_0 = ((__icse_7 + __icse_9) + __icse_9) + (((i_k - 1) * __icse_8 + (i_j - 1)) + 1)
                        branch_stack[__idx_branch_stack_0] = 0
                    end
                    u[i_j] = 0.5 * (rhs[i_j] + left * right + left)
                    __icse_10 = ((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_j - 1)) + 1
                    __idx_left_stack_5 = __icse_10
                    left_stack[__idx_left_stack_5] = left
                    __idx_right_stack_7 = __icse_10
                    right_stack[__idx_right_stack_7] = right
                else
                    __idx_branch_stack_0 = max(0, div(i_nu - 1, 1) + 1) + (((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_0] = 0
                end
            end
        else
            __idx_branch_stack_0 = (i_k - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
    end
    for i_k = i_nu:-1:1
        __idx_branch_stack_0 = (i_k - 1) + 1
        __branch = branch_stack[__idx_branch_stack_0]
        if __branch == 1
            for i_j = i_n:-1:1
                __idx_branch_stack_0 = max(0, div(i_nu - 1, 1) + 1) + (((i_k - 1) * (div(i_n - 1, 1) + 1) + (i_j - 1)) + 1)
                __branch = branch_stack[__idx_branch_stack_0]
                if __branch == 1
                    __icse_11 = div(i_n - 1, 1) + 1
                    __icse_12 = ((i_k - 1) * __icse_11 + (i_j - 1)) + 1
                    __idx_left_stack_0 = __icse_12
                    left = left_stack[__idx_left_stack_0]
                    __idx_right_stack_2 = __icse_12
                    right = right_stack[__idx_right_stack_2]
                    __oldb_0 = ub[i_j]
                    ub[i_j] = 0.0
                    __cse_13 = 0.5__oldb_0
                    rhsb[i_j] = rhsb[i_j] + __cse_13
                    leftb = leftb + right * __cse_13
                    rightb = rightb + left * __cse_13
                    leftb = leftb + __cse_13
                    __icse_14 = max(0, div(i_nu - 1, 1) + 1)
                    __icse_15 = __icse_14 * max(0, __icse_11)
                    __idx_branch_stack_10 = ((__icse_14 + __icse_15) + __icse_15) + __icse_12
                    __branch = branch_stack[__idx_branch_stack_10]
                    if __branch == 1
                        __oldb_0 = rightb
                        rightb = 0.0
                        ub[i_j + 1] = ub[i_j + 1] + __oldb_0
                    end
                    rightb = 0.0
                    __icse_16 = max(0, div(i_nu - 1, 1) + 1)
                    __icse_17 = div(i_n - 1, 1) + 1
                    __idx_branch_stack_14 = (__icse_16 + __icse_16 * max(0, __icse_17)) + (((i_k - 1) * __icse_17 + (i_j - 1)) + 1)
                    __branch = branch_stack[__idx_branch_stack_14]
                    if __branch == 1
                        __oldb_0 = leftb
                        leftb = 0.0
                        ub[i_j - 1] = ub[i_j - 1] + __oldb_0
                    end
                    leftb = 0.0
                end
            end
        end
    end
    return nothing
end

function nest_if_scalar(u, rhs, i_nu, i_n)
    for i_k = 1:i_nu
        if i_nu >= 0
            for i_j = 1:i_n
                if i_nu >= 0
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
    end
end
