function bilinear_d(loss, lossd, x, xd, a, ad, y, yd, i_m, i_n)
    for idx = 1:i_m * i_n
        i_id = 0.0
        __icse_0 = idx - 1
        i_i = div(__icse_0, i_n) + 1
        i_jd = 0.0
        i_j = mod(__icse_0, i_n) + 1
        __cse_1 = a[i_i, i_j]
        __cse_2 = y[i_j]
        __cse_3 = x[i_i]
        lossd[1] = lossd[1] + (((__cse_1 * __cse_2) * xd[i_i] + (__cse_3 * __cse_2) * ad[i_i, i_j]) + (__cse_3 * __cse_1) * yd[i_j])
        loss[1] = loss[1] + __cse_3 * __cse_1 * __cse_2
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
