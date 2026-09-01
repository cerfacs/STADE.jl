function initstacks_matvec_loss_b()
    return nothing
end

function matvec_loss_hv(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n, lossd, lossbd, ad, abd, ud, ubd, vd, vbd)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
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
    for i_i2 = i_m:-1:1
        __cse_4 = lossb[1]
        __cse_5 = 2 * v[i_i2]
        vbd[i_i2] = vbd[i_i2] + (__cse_4 * (2 * vd[i_i2]) + __cse_5 * lossbd[1])
        vb[i_i2] = vb[i_i2] + __cse_5 * __cse_4
    end
    for idx = i_m * i_n:-1:1
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        __cse_0d = vbd[i_i]
        __cse_0 = vb[i_i]
        __cse_6 = u[i_j]
        abd[i_i, i_j] = abd[i_i, i_j] + (__cse_0 * ud[i_j] + __cse_6 * __cse_0d)
        ab[i_i, i_j] = ab[i_i, i_j] + __cse_6 * __cse_0
        __cse_7 = a[i_i, i_j]
        ubd[i_j] = ubd[i_j] + (__cse_0 * ad[i_i, i_j] + __cse_7 * __cse_0d)
        ub[i_j] = ub[i_j] + __cse_7 * __cse_0
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
