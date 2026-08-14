function weightedsumsq_d(loss, lossd, u, ud, w, wd, i_n)
    for i_seq_x = 1:i_n
        lossd[1] = lossd[1] + (u[i_seq_x] ^ 2 * wd[i_seq_x] + w[i_seq_x] * ((2 * u[i_seq_x]) * ud[i_seq_x]))
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
    return nothing
end

function weightedsumsq(loss, u, w, i_n)
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + w[i_seq_x] * u[i_seq_x] ^ 2
    end
end
