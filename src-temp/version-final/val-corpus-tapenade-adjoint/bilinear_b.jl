function initstacks_bilinear_b()
    return
end
function bilinear_b(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n)
    for i_seq_i = i_m:-1:1
        for i_seq_j = i_n:-1:1
            ab[i_seq_i, i_seq_j] = ab[i_seq_i, i_seq_j] + x[i_seq_i] * y[i_seq_j] * lossb[1]
            tempb = a[i_seq_i, i_seq_j] * lossb[1]
            xb[i_seq_i] = xb[i_seq_i] + y[i_seq_j] * tempb
            yb[i_seq_j] = yb[i_seq_j] + x[i_seq_i] * tempb
        end
    end
    return 
end
function bilinear(loss, x, a, y, i_m, i_n)
    for i_seq_i = 1:i_m
        for i_seq_j = 1:i_n
            loss[1] = loss[1] + x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
        end
    end
end
