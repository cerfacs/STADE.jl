function initstacks_bilinear_b()
    return nothing
end

function bilinear_hv(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n, lossd, lossbd, xd, xbd, ad, abd, yd, ybd)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        __cse_4 = a[i_i, i_j]
        __cse_5 = y[i_j]
        __cse_6 = x[i_i]
        lossd[1] = lossd[1] + (((__cse_4 * __cse_5) * xd[i_i] + (__cse_6 * __cse_5) * ad[i_i, i_j]) + (__cse_6 * __cse_4) * yd[i_j])
        loss[1] = loss[1] + __cse_6 * __cse_4 * __cse_5
    end
    for idx = i_m * i_n:-1:1
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        __cse_0d = ad[i_i, i_j]
        __cse_0 = a[i_i, i_j]
        __cse_1d = yd[i_j]
        __cse_1 = y[i_j]
        __cse_2d = lossbd[1]
        __cse_2 = lossb[1]
        __cse_7 = __cse_0 * __cse_1
        xbd[i_i] = xbd[i_i] + (__cse_2 * (__cse_1 * __cse_0d + __cse_0 * __cse_1d) + __cse_7 * __cse_2d)
        xb[i_i] = xb[i_i] + __cse_7 * __cse_2
        __cse_3d = xd[i_i]
        __cse_3 = x[i_i]
        __cse_8 = __cse_3 * __cse_1
        abd[i_i, i_j] = abd[i_i, i_j] + (__cse_2 * (__cse_1 * __cse_3d + __cse_3 * __cse_1d) + __cse_8 * __cse_2d)
        ab[i_i, i_j] = ab[i_i, i_j] + __cse_8 * __cse_2
        __cse_9 = __cse_3 * __cse_0
        ybd[i_j] = ybd[i_j] + (__cse_2 * (__cse_0 * __cse_3d + __cse_3 * __cse_0d) + __cse_9 * __cse_2d)
        yb[i_j] = yb[i_j] + __cse_9 * __cse_2
    end
    return nothing
end

function bilinear(loss, x, a, y, i_m, i_n)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
    end
end
