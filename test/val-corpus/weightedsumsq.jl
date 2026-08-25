function weightedsumsq(loss, u, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + w[i_x] * u[i_x] ^ 2
    end
end
