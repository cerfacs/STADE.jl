function initstacks_hoist_bins_b()
    return nothing
end

function hoist_bins_b(y, yb, z, zb, x, xb, w, wb, v, vb, i_n, i_nb)
    for i_r = 1:i_n
        i_b = mod(i_r - 1, i_nb) + 1
        for i_e = 1:mod(i_r, 4) + 1
            i_s = (i_r - 1) * 4 + i_e
            __cse_0 = x[i_s]
            y[i_b] = y[i_b] + __cse_0 * w[i_e]
            z[i_s] = __cse_0 * v[i_b]
        end
    end
    for i_r = i_n:-1:1
        i_b = mod(i_r - 1, i_nb) + 1
        for i_e = mod(i_r, 4) + 1:-1:1
            i_s = (i_r - 1) * 4 + i_e
            __oldb_0 = zb[i_s]
            zb[i_s] = 0.0
            xb[i_s] = xb[i_s] + v[i_b] * __oldb_0
            __cse_1 = x[i_s]
            vb[i_b] = vb[i_b] + __cse_1 * __oldb_0
            __cse_2 = yb[i_b]
            xb[i_s] = xb[i_s] + w[i_e] * __cse_2
            wb[i_e] = wb[i_e] + __cse_1 * __cse_2
        end
    end
    return nothing
end

function hoist_bins(y, z, x, w, v, i_n, i_nb)
    for i_r = 1:i_n
        i_b = mod(i_r - 1, i_nb) + 1
        for i_e = 1:mod(i_r, 4) + 1
            i_s = (i_r - 1) * 4 + i_e
            y[i_b] = y[i_b] + x[i_s] * w[i_e]
            z[i_s] = x[i_s] * v[i_b]
        end
    end
end
