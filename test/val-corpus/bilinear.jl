function bilinear(loss, x, a, y, i_m, i_n)
    for i_i = 1:i_m
        for i_j = 1:i_n
            loss[1] = loss[1] + x[i_i] * a[i_i, i_j] * y[i_j]
        end
    end
end
