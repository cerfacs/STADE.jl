function initstacks_matvec_loss_b()
    return nothing
end

function matvec_loss_b(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n)
    for idx = 1:i_m * i_n
        __icse_0 = idx - 1
        i_i = div(__icse_0, i_n) + 1
        i_j = mod(__icse_0, i_n) + 1
        v[i_i] = v[i_i] + a[i_i, i_j] * u[i_j]
    end
    for i_i2 = 1:i_m
        loss[1] = loss[1] + v[i_i2] ^ 2
    end
    for i_i2 = i_m:-1:1
        vb[i_i2] = vb[i_i2] + (2 * v[i_i2]) * lossb[1]
    end
    for idx = i_m * i_n:-1:1
        __icse_1 = idx - 1
        i_i = div(__icse_1, i_n) + 1
        i_j = mod(__icse_1, i_n) + 1
        __cse_2 = vb[i_i]
        ab[i_i, i_j] = ab[i_i, i_j] + u[i_j] * __cse_2
        ub[i_j] = ub[i_j] + a[i_i, i_j] * __cse_2
    end
    return nothing
end

function matvec_loss(loss, a, u, v, i_m, i_n)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        v[i_i] = v[i_i] + a[i_i, i_j] * u[i_j]
    end
    for i_i2 = 1:i_m
        loss[1] = loss[1] + v[i_i2] ^ 2
    end
end
