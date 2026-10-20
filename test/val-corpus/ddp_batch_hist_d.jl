function ddp_batch_hist_d(loss, lossd, u, ud, a, ad, hist, histd, v, vd, i_n, i_k)
    for i_x = 1:i_n
        __cse_0 = u[i_x]
        __cse_1 = a[i_x]
        vd[i_x] = __cse_0 * ad[i_x] + __cse_1 * ud[i_x]
        v[i_x] = __cse_1 * __cse_0
    end
    for i_x2 = 1:i_n
        i_bind = 0.0
        i_bin = mod(i_x2 - 1, i_k) + 1
        __cse_2 = v[i_x2]
        __cse_3 = __cse_2 * vd[i_x2]
        histd[i_bin] = histd[i_bin] + (__cse_3 + __cse_3)
        hist[i_bin] = hist[i_bin] + __cse_2 * __cse_2
    end
    for i_x3 = 1:i_n
        __cse_4 = v[i_x3]
        lossd[1] = lossd[1] + (2__cse_4) * vd[i_x3]
        loss[1] = loss[1] + __cse_4 ^ 2
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
