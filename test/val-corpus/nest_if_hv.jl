function initstacks_nest_if_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function nest_if_hv(x, xb, y, yb, i_n, xd, xbd, yd, ybd, branch_stack)
    for i = 1:i_n
        if i > 1
            __idx_branch_stack_0 = (i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            if i > 2
                __idx_branch_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((i - 1) + 1)
                branch_stack[__idx_branch_stack_0] = 1
                __cse_0d = xd[i]
                __cse_0 = x[i]
                __hcse_0 = __cse_0 * __cse_0d
                yd[i] = yd[i] + (__hcse_0 + __hcse_0)
                y[i] = y[i] + __cse_0 * __cse_0
            else
                __idx_branch_stack_0 = max(0, div(i_n - 1, 1) + 1) + ((i - 1) + 1)
                branch_stack[__idx_branch_stack_0] = 0
                __cse_1d = xd[i]
                __cse_1 = x[i]
                __hcse_1 = (__cse_1 * __cse_1) * __cse_1d
                yd[i] = yd[i] + ((__hcse_1 + __hcse_1) + __hcse_1)
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
                __hcse_2 = yb[i]
                __hcse_3 = x[i]
                __cse_2d = __hcse_2 * xd[i] + __hcse_3 * ybd[i]
                __cse_2 = __hcse_3 * __hcse_2
                xbd[i] = xbd[i] + __cse_2d
                xb[i] = xb[i] + __cse_2
                xbd[i] = xbd[i] + __cse_2d
                xb[i] = xb[i] + __cse_2
            else
                __cse_3d = xd[i]
                __cse_3 = x[i]
                __hcse_4 = yb[i]
                __hcse_5 = __cse_3 * __cse_3d
                __hcse_6 = __cse_3 * __cse_3
                __cse_4d = __hcse_4 * (__hcse_5 + __hcse_5) + __hcse_6 * ybd[i]
                __cse_4 = __hcse_6 * __hcse_4
                xbd[i] = xbd[i] + __cse_4d
                xb[i] = xb[i] + __cse_4
                xbd[i] = xbd[i] + __cse_4d
                xb[i] = xb[i] + __cse_4
                xbd[i] = xbd[i] + __cse_4d
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
