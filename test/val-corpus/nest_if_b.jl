function initstacks_nest_if_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function nest_if_b(x, xb, y, yb, i_n, branch_stack)
    for i = 1:i_n
        if i > 1
            __idx_branch_stack_0 = (i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            if i > 2
                __idx_branch_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((i - 1) + 1)
                branch_stack[__idx_branch_stack_0] = 1
                __cse_0 = x[i]
                y[i] = y[i] + __cse_0 * __cse_0
            else
                __idx_branch_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((i - 1) + 1)
                branch_stack[__idx_branch_stack_0] = 0
                __cse_1 = x[i]
                y[i] = y[i] + __cse_1 * __cse_1 * __cse_1
            end
        else
            __idx_branch_stack_0 = (i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
        end
    end
    for i = i_n:-1:1
        __idx_branch_stack_0 = (i - 1) + 1
        __branch = branch_stack[__idx_branch_stack_0]
        if __branch == 1
            __idx_branch_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((i - 1) + 1)
            __branch = branch_stack[__idx_branch_stack_0]
            if __branch == 1
                __cse_2 = x[i] * yb[i]
                xb[i] = xb[i] + __cse_2
                xb[i] = xb[i] + __cse_2
            else
                __cse_3 = x[i]
                __cse_4 = (__cse_3 * __cse_3) * yb[i]
                xb[i] = xb[i] + __cse_4
                xb[i] = xb[i] + __cse_4
                xb[i] = xb[i] + __cse_4
            end
        end
    end
    return nothing
end

function nest_if(x, y, i_n)
    for i = 1:i_n
        if i > 1
            if i > 2
                y[i] = y[i] + x[i] * x[i]
            else
                y[i] = y[i] + x[i] * x[i] * x[i]
            end
        end
    end
end
