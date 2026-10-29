function initstacks_ii_multiversion_b(i_n)
    return nothing
end

function ii_multiversion_b(x, xb, w, wb, y, yb, loss, lossb, i_n, i_nc)
    a = 0.0
    wv = 0.0
    ab = 0.0
    wvb = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[(ci - 1) * 3 + 1]
            a = a + x[t] * wv
            wv = w[(ci - 1) * 3 + 2]
            a = a + x[t + 1] * wv
            wv = w[(ci - 1) * 3 + 3]
            a = a + x[t + 2] * wv
        end
        y[t] = a
    end
    for k = 1:i_n
        __cse_0 = y[k]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for k = i_n:-1:1
        __cse_1 = y[k] * lossb[1]
        yb[k] = yb[k] + __cse_1
        yb[k] = yb[k] + __cse_1
    end
    for t = i_n:-1:1
        __oldb_0 = yb[t]
        yb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            wv = w[(ci - 1) * 3 + 3]
            xb[t + 2] = xb[t + 2] + wv * ab
            wvb = wvb + x[t + 2] * ab
            wv = w[(ci - 1) * 3 + 2]
            __oldb_1 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 3] = wb[(ci - 1) * 3 + 3] + __oldb_1
            xb[t + 1] = xb[t + 1] + wv * ab
            wvb = wvb + x[t + 1] * ab
            wv = w[(ci - 1) * 3 + 1]
            __oldb_1 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 2] = wb[(ci - 1) * 3 + 2] + __oldb_1
            xb[t] = xb[t] + wv * ab
            wvb = wvb + x[t] * ab
            __oldb_0 = wvb
            wvb = 0.0
            wb[(ci - 1) * 3 + 1] = wb[(ci - 1) * 3 + 1] + __oldb_0
        end
        ab = 0.0
    end
    return nothing
end

function ii_multiversion(x, w, y, loss, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[(ci - 1) * 3 + 1]
            a = a + x[t] * wv
            wv = w[(ci - 1) * 3 + 2]
            a = a + x[t + 1] * wv
            wv = w[(ci - 1) * 3 + 3]
            a = a + x[t + 2] * wv
        end
        y[t] = a
    end
    for k = 1:i_n
        loss[1] = loss[1] + y[k] * y[k]
    end
end
