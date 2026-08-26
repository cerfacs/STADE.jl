function initstacks_matvec_loss_b()
    return nothing
end

function matvec_loss_hv(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n, lossd, lossbd, ad, abd, ud, ubd, vd, vbd)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        vd[i_i] = vd[i_i] + (u[i_j] * ad[i_i, i_j] + a[i_i, i_j] * ud[i_j])
        v[i_i] = v[i_i] + a[i_i, i_j] * u[i_j]
    end
    for i_i2 = 1:i_m
        lossd[1] = lossd[1] + (2 * v[i_i2]) * vd[i_i2]
        loss[1] = loss[1] + v[i_i2] ^ 2
    end
    for i_i2 = i_m:-1:1
        vbd[i_i2] = vbd[i_i2] + (lossb[1] * (2 * vd[i_i2]) + (2 * v[i_i2]) * lossbd[1])
        vb[i_i2] = vb[i_i2] + (2 * v[i_i2]) * lossb[1]
    end
    for idx = i_m * i_n:-1:1
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        abd[i_i, i_j] = abd[i_i, i_j] + (vb[i_i] * ud[i_j] + u[i_j] * vbd[i_i])
        ab[i_i, i_j] = ab[i_i, i_j] + u[i_j] * vb[i_i]
        ubd[i_j] = ubd[i_j] + (vb[i_i] * ad[i_i, i_j] + a[i_i, i_j] * vbd[i_i])
        ub[i_j] = ub[i_j] + a[i_i, i_j] * vb[i_i]
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
