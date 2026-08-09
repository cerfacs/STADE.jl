function initstacks_weightedsumsq_b()
    return nothing
end

function weightedsumsq_b(loss, lossb, u, ub, w, wb, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + u[i_seq_x] ^ 2 * lossb[1]
        ub[i_seq_x] = ub[i_seq_x] + (2 * u[i_seq_x]) * (w[i_seq_x] * lossb[1])
    end
    return nothing
end

function weightedsumsq(loss, u, w, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_seq_x = 1:i_n
        #= none:3 =#
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
        #= none:4 =#
    end
end
