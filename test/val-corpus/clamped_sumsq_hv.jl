function initstacks_clamped_sumsq_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function clamped_sumsq_hv(loss, lossb, u, ub, i_n, lossd, lossbd, ud, ubd, branch_stack)
    w = 0.0
    wb = 0.0
    wd = 0.0
    wbd = 0.0
    for i_x = 1:i_n
        if u[i_x] > 0.0
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            __hcse_0 = u[i_x]
            wd = (2__hcse_0) * ud[i_x]
            w = __hcse_0 ^ 2
        else
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
            wd = 0.0
            w = 0.0
        end
        lossd[1] = lossd[1] + wd
        loss[1] = loss[1] + w
    end
    for i_x = i_n:-1:1
        wbd = wbd + lossbd[1]
        wb = wb + lossb[1]
        __idx_branch_stack_1 = (i_x - 1) + 1
        __branch = branch_stack[__idx_branch_stack_1]
        if __branch == 1
            __oldb_0d = wbd
            __oldb_0 = wb
            wbd = 0.0
            wb = 0.0
            __hcse_1 = 2 * u[i_x]
            ubd[i_x] = ubd[i_x] + (__oldb_0 * (2 * ud[i_x]) + __hcse_1 * __oldb_0d)
            ub[i_x] = ub[i_x] + __hcse_1 * __oldb_0
        else
            wbd = 0.0
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
