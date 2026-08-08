function two_field_loss_d(loss, lossd, u, ud, v, vd, p, pd, q, qd, i_n)
    for i_x = 1:i_n
        pd[i_x] = (2 * u[i_x]) * ud[i_x]
        p[i_x] = u[i_x] ^ 2
    end
    for i_x = 1:i_n
        qd[i_x] = (3 * v[i_x] ^ 2) * vd[i_x]
        q[i_x] = v[i_x] ^ 3
    end
    for i_seq_x = 1:i_n
        lossd[1] = (lossd[1] + pd[i_seq_x]) + qd[i_seq_x]
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
    end
    return nothing
end

function two_field_loss(loss, u, v, p, q, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        p[i_x] = u[i_x] ^ 2
        #= none:4 =#
    end
    #= none:5 =#
    for i_x = 1:i_n
        #= none:6 =#
        q[i_x] = v[i_x] ^ 3
        #= none:7 =#
    end
    #= none:8 =#
    for i_seq_x = 1:i_n
        #= none:9 =#
        loss[1] = loss[1] + p[i_seq_x] + q[i_seq_x]
        #= none:10 =#
    end
end
