function initstacks_ddp_batch_hist_b()
    return nothing
end

function ddp_batch_hist_b(loss, lossb, u, ub, a, ab, hist, histb, v, vb, i_n, i_k)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        i_bin = mod(i_x2 - 1, i_k) + 1
        __cse_0 = v[i_x2]
        hist[i_bin] = hist[i_bin] + __cse_0 * __cse_0
    end
    for i_x3 = 1:i_n
        loss[1] = loss[1] + v[i_x3] ^ 2
    end
    for i_x3 = i_n:-1:1
        vb[i_x3] = vb[i_x3] + (2 * v[i_x3]) * lossb[1]
    end
    for i_x2 = i_n:-1:1
        i_bin = mod(i_x2 - 1, i_k) + 1
        __cse_1 = v[i_x2] * histb[i_bin]
        vb[i_x2] = vb[i_x2] + __cse_1
        vb[i_x2] = vb[i_x2] + __cse_1
    end
    for i_x = i_n:-1:1
        __oldb_0 = vb[i_x]
        vb[i_x] = 0.0
        ab[i_x] = ab[i_x] + u[i_x] * __oldb_0
        ub[i_x] = ub[i_x] + a[i_x] * __oldb_0
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
