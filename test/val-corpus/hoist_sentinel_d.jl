function hoist_sentinel_d(y, yd, x, xd, w, wd, i_n)
    for i_r = 1:i_n
        i_bd = 0.0
        i_b = mod(i_r, 3)
        for i_e = 1:i_b
            __cse_0 = w[i_e]
            __cse_1 = x[(i_r - 1) * 2 + i_e]
            yd[i_b] = yd[i_b] + (__cse_0 * xd[(i_r - 1) * 2 + i_e] + __cse_1 * wd[i_e])
            y[i_b] = y[i_b] + __cse_1 * __cse_0
        end
    end
    return nothing
end

function hoist_sentinel(y, x, w, i_n)
    for i_r = 1:i_n
        i_b = mod(i_r, 3)
        for i_e = 1:i_b
            y[i_b] = y[i_b] + x[(i_r - 1) * 2 + i_e] * w[i_e]
        end
    end
end
