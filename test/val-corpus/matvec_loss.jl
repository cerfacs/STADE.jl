function matvec_loss(loss, a, u, v, i_m, i_n)
    for i_i = 1:i_m
        for i_j = 1:i_n
            v[i_i] = v[i_i] + a[i_i, i_j] * u[i_j]
        end
    end
    for i_i2 = 1:i_m
        loss[1] = loss[1] + v[i_i2] ^ 2
    end
end
