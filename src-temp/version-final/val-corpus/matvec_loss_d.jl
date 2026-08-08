function matvec_loss_d(loss, lossd, a, ad, u, ud, v, vd, i_m, i_n)
    for i_i = 1:i_m
        for i_seq_j = 1:i_n
            vd[i_i] = vd[i_i] + (u[i_seq_j] * ad[i_i, i_seq_j] + a[i_i, i_seq_j] * ud[i_seq_j])
            v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
        end
    end
    for i_seq_i = 1:i_m
        lossd[1] = lossd[1] + (2 * v[i_seq_i]) * vd[i_seq_i]
        loss[1] = loss[1] + v[i_seq_i] ^ 2
    end
    return nothing
end

function matvec_loss(loss, a, u, v, i_m, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_i = 1:i_m
        #= none:3 =#
        for i_seq_j = 1:i_n
            #= none:4 =#
            v[i_i] = v[i_i] + a[i_i, i_seq_j] * u[i_seq_j]
            #= none:5 =#
        end
        #= none:6 =#
    end
    #= none:7 =#
    for i_seq_i = 1:i_m
        #= none:8 =#
        loss[1] = loss[1] + v[i_seq_i] ^ 2
        #= none:9 =#
    end
end
