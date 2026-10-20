function initstacks_idm_overlap_plain_b()
    return nothing
end

function idm_overlap_plain_hv(a, ab, x, xb, y, yb, loss, lossb, i_n, ad, abd, xd, xbd, yd, ybd, lossd, lossbd)
    for i_t = 1:i_n
        ad[i_t] = 2.0 * xd[i_t]
        a[i_t] = x[i_t] * 2.0
        ad[i_t + 1] = 3.0 * yd[i_t]
        a[i_t + 1] = y[i_t] * 3.0
    end
    for i_k = 1:i_n
        __cse_0d = ad[i_k]
        __cse_0 = a[i_k]
        __hcse_0 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_0 + __hcse_0)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for i_k = i_n:-1:1
        __hcse_1 = lossb[1]
        __hcse_2 = a[i_k]
        __cse_1d = __hcse_1 * ad[i_k] + __hcse_2 * lossbd[1]
        __cse_1 = __hcse_2 * __hcse_1
        abd[i_k] = abd[i_k] + __cse_1d
        ab[i_k] = ab[i_k] + __cse_1
        abd[i_k] = abd[i_k] + __cse_1d
        ab[i_k] = ab[i_k] + __cse_1
    end
    for i_t = i_n:-1:1
        __oldb_0d = abd[i_t + 1]
        __oldb_0 = ab[i_t + 1]
        abd[i_t + 1] = 0.0
        ab[i_t + 1] = 0.0
        ybd[i_t] = ybd[i_t] + 3.0__oldb_0d
        yb[i_t] = yb[i_t] + 3.0__oldb_0
        __oldb_0d = abd[i_t]
        __oldb_0 = ab[i_t]
        abd[i_t] = 0.0
        ab[i_t] = 0.0
        xbd[i_t] = xbd[i_t] + 2.0__oldb_0d
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
