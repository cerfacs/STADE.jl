function initstacks_affine_loss_b()
    return nothing
end

function affine_loss_hv(loss, lossb, u, ub, a, ab, b, bb, v, vb, i_n, lossd, lossbd, ud, ubd, ad, abd, bd, bbd, vd, vbd)
    for i_x = 1:i_n
        __hcse_0 = u[i_x]
        __hcse_1 = a[i_x]
        vd[i_x] = (__hcse_0 * ad[i_x] + __hcse_1 * ud[i_x]) + bd[i_x]
        v[i_x] = __hcse_1 * __hcse_0 + b[i_x]
    end
    for i_x2 = 1:i_n
        __hcse_2 = v[i_x2]
        lossd[1] = lossd[1] + (2__hcse_2) * vd[i_x2]
        loss[1] = loss[1] + __hcse_2 ^ 2
    end
    for i_x2 = i_n:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = 2 * v[i_x2]
        vbd[i_x2] = vbd[i_x2] + (__hcse_3 * (2 * vd[i_x2]) + __hcse_4 * lossbd[1])
        vb[i_x2] = vb[i_x2] + __hcse_4 * __hcse_3
    end
    for i_x = i_n:-1:1
        __oldb_0d = vbd[i_x]
        __oldb_0 = vb[i_x]
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
        __hcse_5 = u[i_x]
        abd[i_x] = abd[i_x] + (__oldb_0 * ud[i_x] + __hcse_5 * __oldb_0d)
        ab[i_x] = ab[i_x] + __hcse_5 * __oldb_0
        __hcse_6 = a[i_x]
        ubd[i_x] = ubd[i_x] + (__oldb_0 * ad[i_x] + __hcse_6 * __oldb_0d)
        ub[i_x] = ub[i_x] + __hcse_6 * __oldb_0
        bbd[i_x] = bbd[i_x] + __oldb_0d
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
