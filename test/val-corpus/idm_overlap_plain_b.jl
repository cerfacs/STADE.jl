function initstacks_idm_overlap_plain_b()
    return nothing
end

function idm_overlap_plain_b(a, ab, x, xb, y, yb, loss, lossb, i_n)
    for i_t = 1:i_n
        a[i_t] = x[i_t] * 2.0
        a[i_t + 1] = y[i_t] * 3.0
    end
    for i_k = 1:i_n
        __cse_0 = a[i_k]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for i_k = i_n:-1:1
        __cse_1 = a[i_k] * lossb[1]
        ab[i_k] = ab[i_k] + __cse_1
        ab[i_k] = ab[i_k] + __cse_1
    end
    for i_t = i_n:-1:1
        __oldb_0 = ab[i_t + 1]
        ab[i_t + 1] = 0.0
        yb[i_t] = yb[i_t] + 3.0__oldb_0
        __oldb_0 = ab[i_t]
        ab[i_t] = 0.0
        xb[i_t] = xb[i_t] + 2.0__oldb_0
    end
    return nothing
end

function idm_overlap_plain(a, x, y, loss, i_n)
    for i_t = 1:i_n
        a[i_t] = x[i_t] * 2.0
        a[i_t + 1] = y[i_t] * 3.0
    end
    for i_k = 1:i_n
        loss[1] = loss[1] + a[i_k] * a[i_k]
    end
end
