function initstacks_cond_loop_choice_b()
    branch_stack = Vector{Int64}(undef, 1)
    return branch_stack
end

function cond_loop_choice_hv(loss, lossb, u, ub, v, vb, i_branch, i_n, lossd, lossbd, ud, ubd, vd, vbd, branch_stack)
    if i_branch == 1
        branch_stack[1] = 1
        for i_x = 1:i_n
            lossd[1] = lossd[1] + (2 * u[i_x]) * ud[i_x]
            loss[1] = loss[1] + u[i_x] ^ 2
        end
    else
        branch_stack[1] = 0
        for i_x = 1:i_n
            lossd[1] = lossd[1] + (2 * v[i_x]) * vd[i_x]
            loss[1] = loss[1] + v[i_x] ^ 2
        end
    end
    __branch = branch_stack[1]
    if __branch == 1
        for i_x = i_n:-1:1
            ubd[i_x] = ubd[i_x] + (lossb[1] * (2 * ud[i_x]) + (2 * u[i_x]) * lossbd[1])
            ub[i_x] = ub[i_x] + (2 * u[i_x]) * lossb[1]
        end
    else
        for i_x = i_n:-1:1
            vbd[i_x] = vbd[i_x] + (lossb[1] * (2 * vd[i_x]) + (2 * v[i_x]) * lossbd[1])
            vb[i_x] = vb[i_x] + (2 * v[i_x]) * lossb[1]
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
