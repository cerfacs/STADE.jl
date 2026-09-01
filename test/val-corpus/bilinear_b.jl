function initstacks_bilinear_b()
    return nothing
end

function bilinear_b(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
    end
    for idx = i_m * i_n:-1:1
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        __cse_0 = a[i_i, i_j]
        __cse_1 = y[i_j]
        __cse_2 = lossb[1]
        xb[i_i] = xb[i_i] + (__cse_0 * __cse_1) * __cse_2
        __cse_3 = x[i_i]
        ab[i_i, i_j] = ab[i_i, i_j] + (__cse_3 * __cse_1) * __cse_2
        yb[i_j] = yb[i_j] + (__cse_3 * __cse_0) * __cse_2
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
