function gather_alias_d(y, yd, x, xd, i_idx, i_n)
    for i_r = 1:i_n
        i_kd = 0.0
        i_k = i_idx[i_r]
        __cse_0 = x[i_r]
        __cse_1 = y[i_k]
        yd[i_r] = __cse_0 * yd[i_k] + __cse_1 * xd[i_r]
        y[i_r] = __cse_1 * __cse_0
    end
    return nothing
end

function gather_alias(y, x, i_idx, i_n)
    for i_r = 1:i_n
        i_k = i_idx[i_r]
        y[i_r] = y[i_k] * x[i_r]
    end
end
