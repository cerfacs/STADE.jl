function initstacks_hoist_bins_b()
    return nothing
end

function hoist_bins_hv(y, yb, z, zb, x, xb, w, wb, v, vb, i_n, i_nb, yd, ybd, zd, zbd, xd, xbd, wd, wbd, vd, vbd)
    for i_r = 1:i_n
        i_b = mod(i_r - 1, i_nb) + 1
        for i_e = 1:mod(i_r, 4) + 1
            i_s = (i_r - 1) * 4 + i_e
            __cse_0d = xd[i_s]
            __cse_0 = x[i_s]
            __hcse_0 = w[i_e]
            yd[i_b] = yd[i_b] + (__hcse_0 * __cse_0d + __cse_0 * wd[i_e])
            y[i_b] = y[i_b] + __cse_0 * __hcse_0
            __hcse_1 = v[i_b]
            zd[i_s] = __hcse_1 * __cse_0d + __cse_0 * vd[i_b]
            z[i_s] = __cse_0 * __hcse_1
        end
    end
    for i_r = i_n:-1:1
        i_b = mod(i_r - 1, i_nb) + 1
        for i_e = mod(i_r, 4) + 1:-1:1
            i_s = (i_r - 1) * 4 + i_e
            __oldb_0d = zbd[i_s]
            __oldb_0 = zb[i_s]
            zbd[i_s] = 0.0
            zb[i_s] = 0.0
            __hcse_2 = v[i_b]
            xbd[i_s] = xbd[i_s] + (__oldb_0 * vd[i_b] + __hcse_2 * __oldb_0d)
            xb[i_s] = xb[i_s] + __hcse_2 * __oldb_0
            __cse_1d = xd[i_s]
            __cse_1 = x[i_s]
            vbd[i_b] = vbd[i_b] + (__oldb_0 * __cse_1d + __cse_1 * __oldb_0d)
            vb[i_b] = vb[i_b] + __cse_1 * __oldb_0
            __cse_2d = ybd[i_b]
            __cse_2 = yb[i_b]
            __hcse_3 = w[i_e]
            xbd[i_s] = xbd[i_s] + (__cse_2 * wd[i_e] + __hcse_3 * __cse_2d)
            xb[i_s] = xb[i_s] + __hcse_3 * __cse_2
            wbd[i_e] = wbd[i_e] + (__cse_2 * __cse_1d + __cse_1 * __cse_2d)
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
