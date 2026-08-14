function relu_field_d(loss, lossd, u, ud, v, vd, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            vd[i_x] = (2 * u[i_x]) * ud[i_x]
            v[i_x] = u[i_x] ^ 2
        else
            vd[i_x] = 0.0
            v[i_x] = 0.0
        end
    end
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + vd[i_seq_x]
        loss[1] = loss[1] + v[i_seq_x]
    end
    return nothing
end

function relu_field(loss, u, v, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            v[i_x] = u[i_x] ^ 2
        else
            v[i_x] = 0.0
        end
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x]
    end
end
