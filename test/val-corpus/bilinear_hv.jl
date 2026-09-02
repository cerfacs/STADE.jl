function initstacks_bilinear_b()
    return nothing
end

function bilinear_hv(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n, lossd, lossbd, xd, xbd, ad, abd, yd, ybd)
    for idx = 1:i_m * i_n
        __icse_0 = idx - 1
        i_i = div(__icse_0, i_n) + 1
        i_j = mod(__icse_0, i_n) + 1
        __hcse_0 = a[i_i, i_j]
        __hcse_1 = y[i_j]
        __hcse_2 = x[i_i]
        lossd[1] = lossd[1] + (((__hcse_0 * __hcse_1) * xd[i_i] + (__hcse_2 * __hcse_1) * ad[i_i, i_j]) + (__hcse_2 * __hcse_0) * yd[i_j])
        loss[1] = loss[1] + __hcse_2 * __hcse_0 * __hcse_1
    end
    for idx = i_m * i_n:-1:1
        __icse_1 = idx - 1
        i_i = div(__icse_1, i_n) + 1
        i_j = mod(__icse_1, i_n) + 1
        __cse_2d = ad[i_i, i_j]
        __cse_2 = a[i_i, i_j]
        __cse_3d = yd[i_j]
        __cse_3 = y[i_j]
        __cse_4d = lossbd[1]
        __cse_4 = lossb[1]
        __hcse_3 = __cse_2 * __cse_3
        xbd[i_i] = xbd[i_i] + (__cse_4 * (__cse_3 * __cse_2d + __cse_2 * __cse_3d) + __hcse_3 * __cse_4d)
        xb[i_i] = xb[i_i] + __hcse_3 * __cse_4
        __cse_5d = xd[i_i]
        __cse_5 = x[i_i]
        __hcse_4 = __cse_5 * __cse_3
        abd[i_i, i_j] = abd[i_i, i_j] + (__cse_4 * (__cse_3 * __cse_5d + __cse_5 * __cse_3d) + __hcse_4 * __cse_4d)
        ab[i_i, i_j] = ab[i_i, i_j] + __hcse_4 * __cse_4
        __hcse_5 = __cse_5 * __cse_2
        ybd[i_j] = ybd[i_j] + (__cse_4 * (__cse_2 * __cse_5d + __cse_5 * __cse_2d) + __hcse_5 * __cse_4d)
        yb[i_j] = yb[i_j] + __hcse_5 * __cse_4
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
