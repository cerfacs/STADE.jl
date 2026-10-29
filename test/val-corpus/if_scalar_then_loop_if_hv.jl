function initstacks_if_scalar_then_loop_if_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1) * max(0, div(2 - 1, 1) + 1))
    left_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1))
    return (branch_stack, left_stack)
end

function if_scalar_then_loop_if_hv(u, ub, x, xb, y, yb, rhs, rhsb, i_n, ud, ubd, xd, xbd, yd, ybd, rhsd, rhsbd, branch_stack, left_stack)
    left_stack_d = Vector{Float64}(undef, length(left_stack))
    left = 0.0
    leftb = 0.0
    leftd = 0.0
    leftbd = 0.0
    for i = 1:i_n
        leftd = 0.0
        left = 0.0
        if i > 1
            __idx_branch_stack_0 = (i - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            leftd = ud[i - 1]
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
                __cse_1d = xd[j]
                __cse_1 = x[j]
                __hcse_0 = __cse_1 * __cse_1d
                yd[i] = yd[i] + (__hcse_0 + __hcse_0)
                y[i] = y[i] + __cse_1 * __cse_1
            else
                __icse_2 = div(i_n - 1, 1) + 1
                __idx_branch_stack_0 = max(0, __icse_2) + (((i - 1) + (j - 1) * __icse_2) + 1)
                branch_stack[__idx_branch_stack_0] = 0
            end
        end
        __hcse_1 = rhs[i]
        __hcse_2 = (__hcse_1 * left) * leftd
        ud[i] = ud[i] + (((left * left) * rhsd[i] + __hcse_2) + __hcse_2)
        u[i] = u[i] + __hcse_1 * left * left
        __idx_left_stack_4 = (i - 1) + 1
        left_stack_d[__idx_left_stack_4] = leftd
        left_stack[__idx_left_stack_4] = left
    end
    for i = i_n:-1:1
        __idx_left_stack_0 = (i - 1) + 1
        leftd = left_stack_d[__idx_left_stack_0]
        left = left_stack[__idx_left_stack_0]
        __cse_3d = ubd[i]
        __cse_3 = ub[i]
        __hcse_3 = left * leftd
        __hcse_4 = left * left
        rhsbd[i] = rhsbd[i] + (__cse_3 * (__hcse_3 + __hcse_3) + __hcse_4 * __cse_3d)
        rhsb[i] = rhsb[i] + __hcse_4 * __cse_3
        __hcse_5 = rhs[i]
        __hcse_6 = __hcse_5 * left
        __cse_4d = __cse_3 * (left * rhsd[i] + __hcse_5 * leftd) + __hcse_6 * __cse_3d
        __cse_4 = __hcse_6 * __cse_3
        leftbd = leftbd + __cse_4d
        leftb = leftb + __cse_4
        leftbd = leftbd + __cse_4d
        leftb = leftb + __cse_4
        for j = 2:-1:1
            __icse_5 = div(i_n - 1, 1) + 1
            __idx_branch_stack_0 = max(0, __icse_5) + (((i - 1) + (j - 1) * __icse_5) + 1)
            __branch = branch_stack[__idx_branch_stack_0]
            if __branch == 1
                __hcse_7 = yb[i]
                __hcse_8 = x[j]
                __cse_6d = __hcse_7 * xd[j] + __hcse_8 * ybd[i]
                __cse_6 = __hcse_8 * __hcse_7
                xbd[j] = xbd[j] + __cse_6d
                xb[j] = xb[j] + __cse_6
                xbd[j] = xbd[j] + __cse_6d
                xb[j] = xb[j] + __cse_6
            end
        end
        __idx_branch_stack_6 = (i - 1) + 1
        __branch = branch_stack[__idx_branch_stack_6]
        if __branch == 1
            __oldb_0d = leftbd
            __oldb_0 = leftb
            leftbd = 0.0
            leftb = 0.0
            ubd[i - 1] = ubd[i - 1] + __oldb_0d
            ub[i - 1] = ub[i - 1] + __oldb_0
        end
        leftbd = 0.0
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
