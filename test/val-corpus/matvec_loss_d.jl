function matvec_loss_d(loss, lossd, a, ad, u, ud, v, vd, i_m, i_n)
    for idx = 1:i_m * i_n
        i_id = 0.0
        __icse_0 = idx - 1
        i_i = div(__icse_0, i_n) + 1
        i_jd = 0.0
        i_j = mod(__icse_0, i_n) + 1
        __cse_1 = u[i_j]
        __cse_2 = a[i_i, i_j]
        vd[i_i] = vd[i_i] + (__cse_1 * ad[i_i, i_j] + __cse_2 * ud[i_j])
        v[i_i] = v[i_i] + __cse_2 * __cse_1
    end
    for i_i2 = 1:i_m
        __cse_3 = v[i_i2]
        lossd[1] = lossd[1] + (2__cse_3) * vd[i_i2]
        loss[1] = loss[1] + __cse_3 ^ 2
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
