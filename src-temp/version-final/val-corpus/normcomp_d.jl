function normcomp_d(loss, lossd, u, ud, v, vd, w, wd, i_n)
    for i_x = 1:i_n
        wd[i_x] = ud[i_x] + -(vd[i_x])
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * w[i_seq_x]) * wd[i_seq_x]
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    return nothing
end

function normcomp(loss, u, v, w, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        w[i_x] = u[i_x] - v[i_x]
        #= none:4 =#
    end
    #= none:5 =#
    for i_seq_x = 1:i_n
        #= none:6 =#
        loss[1] = loss[1] + w[i_seq_x] ^ 2
        #= none:7 =#
    end
end
