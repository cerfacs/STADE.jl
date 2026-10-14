function initstacks_ddp_branch_b(i_n)
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1))
    return branch_stack
end

function ddp_branch_hv(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, bd, bbd, vd, vbd, branch_stack)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 1
            __hcse_0 = u[i_x]
            __hcse_1 = a[i_x]
            vd[i_x] = __hcse_0 * ad[i_x] + __hcse_1 * ud[i_x]
            v[i_x] = __hcse_1 * __hcse_0
        else
            __idx_branch_stack_0 = (i_x - 1) + 1
            branch_stack[__idx_branch_stack_0] = 0
            __cse_0d = ud[i_x]
            __cse_0 = u[i_x]
            __hcse_2 = b[i_x]
            __hcse_3 = (__hcse_2 * __cse_0) * __cse_0d
            vd[i_x] = ((__cse_0 * __cse_0) * bd[i_x] + __hcse_3) + __hcse_3
            v[i_x] = __hcse_2 * __cse_0 * __cse_0
        end
    end
    for i_x2 = 1:i_n
        __hcse_4 = v[i_x2]
        lossd[1] = lossd[1] + (2__hcse_4) * vd[i_x2]
        loss[1] = loss[1] + __hcse_4 ^ 2
    end
    for i_x2 = i_n:-1:1
        __hcse_5 = lossb[1]
        __hcse_6 = 2 * v[i_x2]
        vbd[i_x2] = vbd[i_x2] + (__hcse_5 * (2 * vd[i_x2]) + __hcse_6 * lossbd[1])
        vb[i_x2] = vb[i_x2] + __hcse_6 * __hcse_5
    end
    for i_x = i_n:-1:1
        __idx_branch_stack_0 = (i_x - 1) + 1
        __branch = branch_stack[__idx_branch_stack_0]
        if __branch == 1
            __oldb_0d = vbd[i_x]
            __oldb_0 = vb[i_x]
            vbd[i_x] = 0.0
            vb[i_x] = 0.0
            __hcse_7 = u[i_x]
            abd[i_x] = abd[i_x] + (__oldb_0 * ud[i_x] + __hcse_7 * __oldb_0d)
            ab[i_x] = ab[i_x] + __hcse_7 * __oldb_0
            __hcse_8 = a[i_x]
            ubd[i_x] = ubd[i_x] + (__oldb_0 * ad[i_x] + __hcse_8 * __oldb_0d)
            ub[i_x] = ub[i_x] + __hcse_8 * __oldb_0
        else
            __oldb_0d = vbd[i_x]
            __oldb_0 = vb[i_x]
            vbd[i_x] = 0.0
            vb[i_x] = 0.0
            __cse_1d = ud[i_x]
            __cse_1 = u[i_x]
            __hcse_9 = __cse_1 * __cse_1d
            __hcse_10 = __cse_1 * __cse_1
            bbd[i_x] = bbd[i_x] + (__oldb_0 * (__hcse_9 + __hcse_9) + __hcse_10 * __oldb_0d)
            bb[i_x] = bb[i_x] + __hcse_10 * __oldb_0
            __hcse_11 = b[i_x]
            __hcse_12 = __hcse_11 * __cse_1
            __cse_2d = __oldb_0 * (__cse_1 * bd[i_x] + __hcse_11 * __cse_1d) + __hcse_12 * __oldb_0d
            __cse_2 = __hcse_12 * __oldb_0
            ubd[i_x] = ubd[i_x] + __cse_2d
            ub[i_x] = ub[i_x] + __cse_2
            ubd[i_x] = ubd[i_x] + __cse_2d
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
