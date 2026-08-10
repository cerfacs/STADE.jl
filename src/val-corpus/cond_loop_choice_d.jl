function cond_loop_choice_d(loss, lossd, u, ud, v, vd, i_branch, i_n)
    if i_branch == 1
        for i_seq_x = 1:i_n
            lossd[1] = lossd[1] + (2 * u[i_seq_x]) * ud[i_seq_x]
            loss[1] = loss[1] + u[i_seq_x] ^ 2
        end
    else
        for i_seq_x = 1:i_n
            lossd[1] = lossd[1] + (2 * v[i_seq_x]) * vd[i_seq_x]
            loss[1] = loss[1] + v[i_seq_x] ^ 2
        end
    end
    return nothing
end

function cond_loop_choice(loss, u, v, i_branch, i_n)
    #= none:1 =#
    #= none:2 =#
    if i_branch == 1
        #= none:3 =#
        for i_seq_x = 1:i_n
            #= none:4 =#
            loss[1] = loss[1] + u[i_seq_x] ^ 2
            #= none:5 =#
        end
    else
        #= none:7 =#
        for i_seq_x = 1:i_n
            #= none:8 =#
            loss[1] = loss[1] + v[i_seq_x] ^ 2
            #= none:9 =#
        end
    end
end
