function bilinear_d(loss, lossd, x, xd, a, ad, y, yd, i_m, i_n)
    for idx = 1:i_m * i_n
        i_id = 0.0
        i_i = div(idx - 1, i_n) + 1
        i_jd = 0.0
        i_j = mod(idx - 1, i_n) + 1
        lossd[1] = lossd[1] + (((a[i_i, i_j] * y[i_j]) * xd[i_i] + (x[i_i] * y[i_j]) * ad[i_i, i_j]) + (x[i_i] * a[i_i, i_j]) * yd[i_j])
        loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
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
