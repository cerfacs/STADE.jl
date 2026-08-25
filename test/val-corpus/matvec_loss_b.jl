function initstacks_matvec_loss_b()
    return nothing
end

function matvec_loss_b(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n)
    for i_i = 1:i_m
        for i_j = 1:i_n
            v[i_i] = v[i_i] + a[i_i, i_j] * u[i_j]
        end
    end
    for i_i2 = 1:i_m
        loss[1] = loss[1] + v[i_i2] ^ 2
    end
    for i_i2 = i_m:-1:1
        vb[i_i2] = vb[i_i2] + (2 * v[i_i2]) * lossb[1]
    end
    for i_i = i_m:-1:1
        for i_j = i_n:-1:1
            ab[i_i, i_j] = ab[i_i, i_j] + u[i_j] * vb[i_i]
            ub[i_j] = ub[i_j] + a[i_i, i_j] * vb[i_i]
        end
    end
    return nothing
end

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
