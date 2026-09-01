function bilinear_d(loss, lossd, x, xd, a, ad, y, yd, i_m, i_n)
    for idx = 1:i_m * i_n
        i_id = 0.0
        i_i = div(idx - 1, i_n) + 1
        i_jd = 0.0
        i_j = mod(idx - 1, i_n) + 1
        __cse_0 = a[i_i, i_j]
        __cse_1 = y[i_j]
        __cse_2 = x[i_i]
        lossd[1] = lossd[1] + (((__cse_0 * __cse_1) * xd[i_i] + (__cse_2 * __cse_1) * ad[i_i, i_j]) + (__cse_2 * __cse_0) * yd[i_j])
        loss[1] = loss[1] + __cse_2 * __cse_0 * __cse_1
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
