function affine_loss_d(loss, lossd, u, ud, a, ad, b, bd, v, vd, i_n)
    for i_x = 1:i_n
        vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (2 * v[i_seq_x]) * vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    return nothing
end

function affine_loss(loss, u, a, b, v, i_n)
    #= none:1 =#
    #= none:2 =#
    for i_x = 1:i_n
        #= none:3 =#
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
        #= none:4 =#
    end
    #= none:5 =#
    for i_seq_x = 1:i_n
        #= none:6 =#
        loss[1] = loss[1] + v[i_seq_x] ^ 2
        #= none:7 =#
    end
end
