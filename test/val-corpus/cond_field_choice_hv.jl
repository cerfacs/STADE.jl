function initstacks_cond_field_choice_b()
    branch_stack = Vector{Int64}(undef, 1)
    return branch_stack
end

function cond_field_choice_hv(loss, lossb, u, ub, v, vb, w, wb, i_branch, i_n, lossd, lossbd, ud, ubd, vd, vbd, wd, wbd, branch_stack)
    if i_branch == 1
        branch_stack[1] = 1
        for i_x = 1:i_n
            __cse_0 = u[i_x]
            wd[i_x] = (2__cse_0) * ud[i_x]
            w[i_x] = __cse_0 ^ 2
        end
    else
        branch_stack[1] = 0
        for i_x = 1:i_n
            __cse_1 = v[i_x]
            wd[i_x] = (2__cse_1) * vd[i_x]
            w[i_x] = __cse_1 ^ 2
        end
    end
    for i_x2 = 1:i_n
        lossd[1] = lossd[1] + wd[i_x2]
        loss[1] = loss[1] + w[i_x2]
    end
    for i_x2 = i_n:-1:1
        wbd[i_x2] = wbd[i_x2] + lossbd[1]
        wb[i_x2] = wb[i_x2] + lossb[1]
    end
    __branch = branch_stack[1]
    if __branch == 1
        for i_x = i_n:-1:1
            __cse_2 = wb[i_x]
            __cse_3 = 2 * u[i_x]
            ubd[i_x] = ubd[i_x] + (__cse_2 * (2 * ud[i_x]) + __cse_3 * wbd[i_x])
            ub[i_x] = ub[i_x] + __cse_3 * __cse_2
            wbd[i_x] = 0.0
            wb[i_x] = 0.0
        end
    else
        for i_x = i_n:-1:1
            __cse_4 = wb[i_x]
            __cse_5 = 2 * v[i_x]
            vbd[i_x] = vbd[i_x] + (__cse_4 * (2 * vd[i_x]) + __cse_5 * wbd[i_x])
            vb[i_x] = vb[i_x] + __cse_5 * __cse_4
            wbd[i_x] = 0.0
            wb[i_x] = 0.0
        end
    end
    return nothing
end

function cond_field_choice(loss, u, v, w, i_branch, i_n)
    if i_branch == 1
        for i_x = 1:i_n
            w[i_x] = u[i_x] ^ 2
        end
    else
        for i_x = 1:i_n
            w[i_x] = v[i_x] ^ 2
        end
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + w[i_x2]
    end
end
