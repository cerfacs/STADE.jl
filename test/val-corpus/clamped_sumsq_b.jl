function initstacks_clamped_sumsq_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function clamped_sumsq_b(loss, lossb, u, ub, i_n, branch_stack)
    w = 0.0
    wb = 0.0
    for i_x = 1:i_n
        if u[i_x] > 0.0
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            w = u[i_x] ^ 2
        else
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
            w = 0.0
        end
        loss[1] = loss[1] + w
    end
    for i_x = i_n:-1:1
        wb = wb + lossb[1]
        __idx_branch_stack_1 = (i_x - 1) + 1
        __branch = branch_stack[__idx_branch_stack_1]
        if __branch == 1
            __oldb_0 = wb
            wb = 0.0
            ub[i_x] = ub[i_x] + (2 * u[i_x]) * __oldb_0
        else
            wb = 0.0
        end
    end
    return nothing
end

function clamped_sumsq(loss, u, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            w = u[i_x] ^ 2
        else
            w = 0.0
        end
        loss[1] = loss[1] + w
    end
end
