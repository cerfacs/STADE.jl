function initstacks_bilinear_b()
    return nothing
end

function bilinear_hv(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n, lossd, lossbd, xd, xbd, ad, abd, yd, ybd)
    for idx = 1:i_m * i_n
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        lossd[1] = lossd[1] + (((a[i_i, i_j] * y[i_j]) * xd[i_i] + (x[i_i] * y[i_j]) * ad[i_i, i_j]) + (x[i_i] * a[i_i, i_j]) * yd[i_j])
        loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
    end
    for idx = i_m * i_n:-1:1
        i_i = div(idx - 1, i_n) + 1
        i_j = mod(idx - 1, i_n) + 1
        xbd[i_i] = xbd[i_i] + (lossb[1] * (y[i_j] * ad[i_i, i_j] + a[i_i, i_j] * yd[i_j]) + (a[i_i, i_j] * y[i_j]) * lossbd[1])
        xb[i_i] = xb[i_i] + (a[i_i, i_j] * y[i_j]) * lossb[1]
        abd[i_i, i_j] = abd[i_i, i_j] + (lossb[1] * (y[i_j] * xd[i_i] + x[i_i] * yd[i_j]) + (x[i_i] * y[i_j]) * lossbd[1])
        ab[i_i, i_j] = ab[i_i, i_j] + (x[i_i] * y[i_j]) * lossb[1]
        ybd[i_j] = ybd[i_j] + (lossb[1] * (a[i_i, i_j] * xd[i_i] + x[i_i] * ad[i_i, i_j]) + (x[i_i] * a[i_i, i_j]) * lossbd[1])
        yb[i_j] = yb[i_j] + (x[i_i] * a[i_i, i_j]) * lossb[1]
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
