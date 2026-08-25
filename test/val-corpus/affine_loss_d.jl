function affine_loss_d(loss, lossd, u, ud, a, ad, b, bd, v, vd, i_n)
    for i_x = 1:i_n
        vd[i_x] = (u[i_x] * ad[i_x] + a[i_x] * ud[i_x]) + bd[i_x]
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x2 = 1:i_n
        lossd[1] = lossd[1] + (2 * v[i_x2]) * vd[i_x2]
        loss[1] = loss[1] + v[i_x2] ^ 2
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
