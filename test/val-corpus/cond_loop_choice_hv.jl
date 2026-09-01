function initstacks_cond_loop_choice_b()
    branch_stack = Vector{Int64}(undef, 1)
    return branch_stack
end

function cond_loop_choice_hv(loss, lossb, u, ub, v, vb, i_branch, i_n, lossd, lossbd, ud, ubd, vd, vbd, branch_stack)
    if i_branch == 1
        branch_stack[1] = 1
        for i_x = 1:i_n
            __cse_0 = u[i_x]
            lossd[1] = lossd[1] + (2__cse_0) * ud[i_x]
            loss[1] = loss[1] + __cse_0 ^ 2
        end
    else
        branch_stack[1] = 0
        for i_x = 1:i_n
            __cse_1 = v[i_x]
            lossd[1] = lossd[1] + (2__cse_1) * vd[i_x]
            loss[1] = loss[1] + __cse_1 ^ 2
        end
    end
    __branch = branch_stack[1]
    if __branch == 1
        for i_x = i_n:-1:1
            __cse_2 = lossb[1]
            __cse_3 = 2 * u[i_x]
            ubd[i_x] = ubd[i_x] + (__cse_2 * (2 * ud[i_x]) + __cse_3 * lossbd[1])
            ub[i_x] = ub[i_x] + __cse_3 * __cse_2
        end
    else
        for i_x = i_n:-1:1
            __cse_4 = lossb[1]
            __cse_5 = 2 * v[i_x]
            vbd[i_x] = vbd[i_x] + (__cse_4 * (2 * vd[i_x]) + __cse_5 * lossbd[1])
            vb[i_x] = vb[i_x] + __cse_5 * __cse_4
        end
    end
    return nothing
end

function cond_loop_choice(loss, u, v, i_branch, i_n)
    if i_branch == 1
        for i_x = 1:i_n
            loss[1] = loss[1] + u[i_x] ^ 2
        end
    else
        for i_x = 1:i_n
            loss[1] = loss[1] + v[i_x] ^ 2
        end
    end
end
