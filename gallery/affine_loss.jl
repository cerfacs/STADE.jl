function affine_loss(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_seq_x = 1:i_n
        loss[1] = loss[1] + v[i_seq_x] ^ 2
    end
end