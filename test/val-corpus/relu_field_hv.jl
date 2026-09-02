function initstacks_relu_field_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function relu_field_hv(loss, lossb, u, ub, v, vb, i_n, lossd, lossbd, ud, ubd, vd, vbd, branch_stack)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            __hcse_0 = u[i_x]
            vd[i_x] = (2__hcse_0) * ud[i_x]
            v[i_x] = __hcse_0 ^ 2
        else
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
            vd[i_x] = 0.0
            v[i_x] = 0.0
        end
    end
    for i_x2 = 1:i_n
        lossd[1] = lossd[1] + vd[i_x2]
        loss[1] = loss[1] + v[i_x2]
    end
    for i_x2 = i_n:-1:1
        vbd[i_x2] = vbd[i_x2] + lossbd[1]
        vb[i_x2] = vb[i_x2] + lossb[1]
    end
    for i_x = i_n:-1:1
        __idx_branch_stack_0 = (i_x - 1) + 1
        __branch = branch_stack[__idx_branch_stack_0]
        if __branch == 1
            __oldb_0d = vbd[i_x]
            __oldb_0 = vb[i_x]
            vbd[i_x] = 0.0
            vb[i_x] = 0.0
            __hcse_1 = 2 * u[i_x]
            ubd[i_x] = ubd[i_x] + (__oldb_0 * (2 * ud[i_x]) + __hcse_1 * __oldb_0d)
            ub[i_x] = ub[i_x] + __hcse_1 * __oldb_0
        else
            vbd[i_x] = 0.0
            vb[i_x] = 0.0
        end
    end
    return nothing
end

function relu_field(loss, u, v, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            v[i_x] = u[i_x] ^ 2
        else
            v[i_x] = 0.0
        end
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2]
    end
end
