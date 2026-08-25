function initstacks_bilinear_b()
    return nothing
end

function bilinear_b(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n)
    for i_i = 1:i_m
        for i_j = 1:i_n
            loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
        end
    end
    for i_i = i_m:-1:1
        for i_j = i_n:-1:1
            xb[i_i] = xb[i_i] + (a[i_i, i_j] * y[i_j]) * lossb[1]
            ab[i_i, i_j] = ab[i_i, i_j] + (x[i_i] * y[i_j]) * lossb[1]
            yb[i_j] = yb[i_j] + (x[i_i] * a[i_i, i_j]) * lossb[1]
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
