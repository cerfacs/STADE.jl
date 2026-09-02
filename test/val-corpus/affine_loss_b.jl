function initstacks_affine_loss_b()
    return nothing
end

function affine_loss_b(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    for i_x2 = i_n:-1:1
        vb[i_x2] = vb[i_x2] + (2 * v[i_x2]) * lossb[1]
    end
    for i_x = i_n:-1:1
        __oldb_0 = vb[i_x]
        vb[i_x] = 0.0
        ab[i_x] = ab[i_x] + u[i_x] * __oldb_0
        ub[i_x] = ub[i_x] + a[i_x] * __oldb_0
        bb[i_x] = bb[i_x] + __oldb_0
    end
    return nothing
end

function affine_loss(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
end
