function initstacks_matvec_loss_b()
    return nothing
end

function matvec_loss_hv(loss, lossb, a, ab, u, ub, v, vb, i_m, i_n, lossd, lossbd, ad, abd, ud, ubd, vd, vbd)
    for idx = 1:i_m * i_n
        __icse_0 = idx - 1
        i_i = div(__icse_0, i_n) + 1
        i_j = mod(__icse_0, i_n) + 1
        __hcse_0 = u[i_j]
        __hcse_1 = a[i_i, i_j]
        vd[i_i] = vd[i_i] + (__hcse_0 * ad[i_i, i_j] + __hcse_1 * ud[i_j])
        v[i_i] = v[i_i] + __hcse_1 * __hcse_0
    end
    for i_i2 = 1:i_m
        __hcse_2 = v[i_i2]
        lossd[1] = lossd[1] + (2__hcse_2) * vd[i_i2]
        loss[1] = loss[1] + __hcse_2 ^ 2
    end
    for i_i2 = i_m:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = 2 * v[i_i2]
        vbd[i_i2] = vbd[i_i2] + (__hcse_3 * (2 * vd[i_i2]) + __hcse_4 * lossbd[1])
        vb[i_i2] = vb[i_i2] + __hcse_4 * __hcse_3
    end
    for idx = i_m * i_n:-1:1
        __icse_1 = idx - 1
        i_i = div(__icse_1, i_n) + 1
        i_j = mod(__icse_1, i_n) + 1
        __cse_2d = vbd[i_i]
        __cse_2 = vb[i_i]
        __hcse_5 = u[i_j]
        abd[i_i, i_j] = abd[i_i, i_j] + (__cse_2 * ud[i_j] + __hcse_5 * __cse_2d)
        ab[i_i, i_j] = ab[i_i, i_j] + __hcse_5 * __cse_2
        __hcse_6 = a[i_i, i_j]
        ubd[i_j] = ubd[i_j] + (__cse_2 * ad[i_i, i_j] + __hcse_6 * __cse_2d)
        ub[i_j] = ub[i_j] + __hcse_6 * __cse_2
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
