function hoist_bins_d(y, yd, z, zd, x, xd, w, wd, v, vd, i_n, i_nb)
    for i_r = 1:i_n
        i_bd = 0.0
        i_b = mod(i_r - 1, i_nb) + 1
        for i_e = 1:mod(i_r, 4) + 1
            i_sd = 0.0
            i_s = (i_r - 1) * 4 + i_e
            __cse_0 = w[i_e]
            __cse_1 = xd[i_s]
            __cse_2 = x[i_s]
            yd[i_b] = yd[i_b] + (__cse_0 * __cse_1 + __cse_2 * wd[i_e])
            y[i_b] = y[i_b] + __cse_2 * __cse_0
            __cse_3 = v[i_b]
            zd[i_s] = __cse_3 * __cse_1 + __cse_2 * vd[i_b]
            z[i_s] = __cse_2 * __cse_3
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
