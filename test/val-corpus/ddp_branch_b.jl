function initstacks_ddp_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function ddp_branch_b(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n, branch_stack)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            v[i_x] = a[i_x] * u[i_x]
        else
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
            __cse_0 = u[i_x]
            v[i_x] = b[i_x] * __cse_0 * __cse_0
        end
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    for i_x2 = i_n:-1:1
        vb[i_x2] = vb[i_x2] + (2 * v[i_x2]) * lossb[1]
    end
    for i_x = i_n:-1:1
        __idx_branch_stack_0 = (i_x - 1) + 1
        __branch = branch_stack[__idx_branch_stack_0]
        if __branch == 1
            __oldb_0 = vb[i_x]
            vb[i_x] = 0.0
            ab[i_x] = ab[i_x] + u[i_x] * __oldb_0
            ub[i_x] = ub[i_x] + a[i_x] * __oldb_0
        else
            __oldb_0 = vb[i_x]
            vb[i_x] = 0.0
            __cse_1 = u[i_x]
            bb[i_x] = bb[i_x] + (__cse_1 * __cse_1) * __oldb_0
            __cse_2 = (b[i_x] * __cse_1) * __oldb_0
            ub[i_x] = ub[i_x] + __cse_2
            ub[i_x] = ub[i_x] + __cse_2
        end
    end
    return nothing
end

function ddp_branch(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            v[i_x] = a[i_x] * u[i_x]
        else
            v[i_x] = b[i_x] * u[i_x] * u[i_x]
        end
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
