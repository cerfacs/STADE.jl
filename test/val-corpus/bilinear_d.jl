function bilinear_d(loss, lossd, x, xd, a, ad, y, yd, i_m, i_n)
    for i_i = 1:i_m
        for i_j = 1:i_n
            lossd[1] = lossd[1] + (((a[i_i, i_j] * y[i_j]) * xd[i_i] + (x[i_i] * y[i_j]) * ad[i_i, i_j]) + (x[i_i] * a[i_i, i_j]) * yd[i_j])
            loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
        end
    end
    return nothing
end

function bilinear(loss, x, a, y, i_m, i_n)
    for i_i = 1:i_m
        for i_j = 1:i_n
            loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
        end
    end
end
