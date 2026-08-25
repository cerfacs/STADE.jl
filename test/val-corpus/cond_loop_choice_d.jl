function cond_loop_choice_d(loss, lossd, u, ud, v, vd, i_branch, i_n)
    if i_branch == 1
        for i_x = 1:i_n
            lossd[1] = lossd[1] + (2 * u[i_x]) * ud[i_x]
            loss[1] = loss[1] + u[i_x] ^ 2
        end
    else
        for i_x = 1:i_n
            lossd[1] = lossd[1] + (2 * v[i_x]) * vd[i_x]
            loss[1] = loss[1] + v[i_x] ^ 2
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
