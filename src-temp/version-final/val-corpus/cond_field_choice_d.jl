function cond_field_choice_d(loss, lossd, u, ud, v, vd, w, wd, i_branch, i_n)
    if i_branch == 1
        for i_x = 1:i_n
            wd[i_x] = (2 * u[i_x]) * ud[i_x]
            w[i_x] = u[i_x] ^ 2
        end
    else
        for i_x = 1:i_n
            wd[i_x] = (2 * v[i_x]) * vd[i_x]
            w[i_x] = v[i_x] ^ 2
        end
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x]
    end
    return nothing
end

function cond_field_choice(loss, u, v, w, i_branch, i_n)
    #= none:1 =#
    #= none:2 =#
    if i_branch == 1
        #= none:3 =#
        for i_x = 1:i_n
            #= none:4 =#
            w[i_x] = u[i_x] ^ 2
            #= none:5 =#
        end
    else
        #= none:7 =#
        for i_x = 1:i_n
            #= none:8 =#
            w[i_x] = v[i_x] ^ 2
            #= none:9 =#
        end
    end
    #= none:11 =#
    for i_seq_x = 1:i_n
        #= none:12 =#
        loss[1] = loss[1] + w[i_seq_x]
        #= none:13 =#
    end
end
