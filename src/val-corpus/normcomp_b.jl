function initstacks_normcomp_b()
    return nothing
end

function normcomp_b(loss, lossb, u, ub, v, vb, w, wb, i_n)
    for i_x = 1:i_n
        w[i_x] = u[i_x] - v[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        wb[i_seq_x] = wb[i_seq_x] + (2 * w[i_seq_x]) * lossb[1]
    end
    for i_x = 1:i_n
        ub[i_x] = ub[i_x] + wb[i_x]
        vb[i_x] = vb[i_x] + -(wb[i_x])
        wb[i_x] = 0.0
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
