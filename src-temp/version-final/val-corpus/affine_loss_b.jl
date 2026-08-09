function initstacks_affine_loss_b()
    return nothing
end

function affine_loss_b(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
    for i_seq_x = i_n:-1:1
        vb[i_seq_x] = vb[i_seq_x] + (2 * v[i_seq_x]) * lossb[1]
    end
    for i_x = 1:i_n
        ab[i_x] = ab[i_x] + u[i_x] * vb[i_x]
        ub[i_x] = ub[i_x] + a[i_x] * vb[i_x]
        bb[i_x] = bb[i_x] + vb[i_x]
        vb[i_x] = 0.0
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
