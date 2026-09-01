function cond_loop_choice_d(loss, lossd, u, ud, v, vd, i_branch, i_n)
    if i_branch == 1
        for i_x = 1:i_n
            __cse_0 = u[i_x]
            lossd[1] = lossd[1] + (2__cse_0) * ud[i_x]
            loss[1] = loss[1] + __cse_0 ^ 2
        end
    else
        for i_x = 1:i_n
            __cse_1 = v[i_x]
            lossd[1] = lossd[1] + (2__cse_1) * vd[i_x]
            loss[1] = loss[1] + __cse_1 ^ 2
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
