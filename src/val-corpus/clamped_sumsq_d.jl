function clamped_sumsq_d(loss, lossd, u, ud, i_n)
    for i_seq_x = 1:i_n
        if u[i_seq_x] > 0.0
            wd = (2 * u[i_seq_x]) * ud[i_seq_x]
            w = u[i_seq_x] ^ 2
        else
            wd = 0.0
            w = 0.0
        end
        lossd[1] = lossd[1] + wd
        loss[1] = loss[1] + w
    end
    return nothing
end

function clamped_sumsq(loss, u, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_seq_x = 1:i_n
        #= none:3 =#
        if u[i_seq_x] > 0.0
            #= none:4 =#
            w = u[i_seq_x] ^ 2
        else
            #= none:6 =#
            w = 0.0
        end
        #= none:8 =#
        loss[1] = loss[1] + w
        #= none:9 =#
    end
end
