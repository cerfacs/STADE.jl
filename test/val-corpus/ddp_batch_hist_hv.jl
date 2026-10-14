function initstacks_ddp_batch_hist_b()
    return nothing
end

function ddp_batch_hist_hv(loss, lossb, u, ub, a, ab, hist, histb, v, vb, i_n, i_k, lossd, lossbd, ud, ubd, ad, abd, histd, histbd, vd, vbd)
    for i_x = 1:i_n
        __hcse_0 = u[i_x]
        __hcse_1 = a[i_x]
        vd[i_x] = __hcse_0 * ad[i_x] + __hcse_1 * ud[i_x]
        v[i_x] = __hcse_1 * __hcse_0
    end
    for i_x2 = 1:i_n
        i_bin = mod(i_x2 - 1, i_k) + 1
        __cse_0d = vd[i_x2]
        __cse_0 = v[i_x2]
        __hcse_2 = __cse_0 * __cse_0d
        histd[i_bin] = histd[i_bin] + (__hcse_2 + __hcse_2)
        hist[i_bin] = hist[i_bin] + __cse_0 * __cse_0
    end
    for i_x3 = 1:i_n
        __hcse_3 = v[i_x3]
        lossd[1] = lossd[1] + (2__hcse_3) * vd[i_x3]
        loss[1] = loss[1] + __hcse_3 ^ 2
    end
    for i_x3 = i_n:-1:1
        __hcse_4 = lossb[1]
        __hcse_5 = 2 * v[i_x3]
        vbd[i_x3] = vbd[i_x3] + (__hcse_4 * (2 * vd[i_x3]) + __hcse_5 * lossbd[1])
        vb[i_x3] = vb[i_x3] + __hcse_5 * __hcse_4
    end
    for i_x2 = i_n:-1:1
        i_bin = mod(i_x2 - 1, i_k) + 1
        __hcse_6 = histb[i_bin]
        __hcse_7 = v[i_x2]
        __cse_1d = __hcse_6 * vd[i_x2] + __hcse_7 * histbd[i_bin]
        __cse_1 = __hcse_7 * __hcse_6
        vbd[i_x2] = vbd[i_x2] + __cse_1d
        vb[i_x2] = vb[i_x2] + __cse_1
        vbd[i_x2] = vbd[i_x2] + __cse_1d
        vb[i_x2] = vb[i_x2] + __cse_1
    end
    for i_x = i_n:-1:1
        __oldb_0d = vbd[i_x]
        __oldb_0 = vb[i_x]
        vbd[i_x] = 0.0
        vb[i_x] = 0.0
        __hcse_8 = u[i_x]
        abd[i_x] = abd[i_x] + (__oldb_0 * ud[i_x] + __hcse_8 * __oldb_0d)
        ab[i_x] = ab[i_x] + __hcse_8 * __oldb_0
        __hcse_9 = a[i_x]
        ubd[i_x] = ubd[i_x] + (__oldb_0 * ad[i_x] + __hcse_9 * __oldb_0d)
        ub[i_x] = ub[i_x] + __hcse_9 * __oldb_0
    end
    return nothing
end

function ddp_batch_hist(loss, u, a, hist, v, i_n, i_k)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        i_bin = mod(i_x2 - 1, i_k) + 1
        hist[i_bin] = hist[i_bin] + v[i_x2] * v[i_x2]
    end
    for i_x3 = 1:i_n
        loss[1] = loss[1] + v[i_x3] ^ 2
    end
end
