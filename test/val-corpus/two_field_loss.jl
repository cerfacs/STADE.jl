function two_field_loss(loss, u, v, p, q, i_n)
    for i_x = 1:i_n
        p[i_x] = u[i_x] ^ 2
    end
    for i_x = 1:i_n
        q[i_x] = v[i_x] ^ 3
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + p[i_x2] + q[i_x2]
    end
end
