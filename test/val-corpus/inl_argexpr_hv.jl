function initstacks_inl_argexpr_b()
    return nothing
end

function inl_argexpr_hv(loss, lossb, w, wb, x, xb, t, tb, i_n, lossd, lossbd, wd, wbd, xd, xbd, td, tbd)
    __inl_stage_1_4 = i_n - 1
    for i_k_stage_c1 = 1:__inl_stage_1_4
        __hcse_0 = x[i_k_stage_c1]
        __hcse_1 = w[i_k_stage_c1]
        td[i_k_stage_c1] = td[i_k_stage_c1] + (__hcse_0 * wd[i_k_stage_c1] + __hcse_1 * xd[i_k_stage_c1])
        t[i_k_stage_c1] = t[i_k_stage_c1] + __hcse_1 * __hcse_0
    end
    for i_j = 1:i_n
        __hcse_2 = t[i_j]
        lossd[1] = lossd[1] + (2__hcse_2) * td[i_j]
        loss[1] = loss[1] + __hcse_2 ^ 2
    end
    __inl_stage_1_4 = i_n - 1
    for i_j = i_n:-1:1
        __hcse_3 = lossb[1]
        __hcse_4 = 2 * t[i_j]
        tbd[i_j] = tbd[i_j] + (__hcse_3 * (2 * td[i_j]) + __hcse_4 * lossbd[1])
        tb[i_j] = tb[i_j] + __hcse_4 * __hcse_3
    end
    for i_k_stage_c1 = __inl_stage_1_4:-1:1
        __cse_0d = tbd[i_k_stage_c1]
        __cse_0 = tb[i_k_stage_c1]
        __hcse_5 = x[i_k_stage_c1]
        wbd[i_k_stage_c1] = wbd[i_k_stage_c1] + (__cse_0 * xd[i_k_stage_c1] + __hcse_5 * __cse_0d)
        wb[i_k_stage_c1] = wb[i_k_stage_c1] + __hcse_5 * __cse_0
        __hcse_6 = w[i_k_stage_c1]
        xbd[i_k_stage_c1] = xbd[i_k_stage_c1] + (__cse_0 * wd[i_k_stage_c1] + __hcse_6 * __cse_0d)
        xb[i_k_stage_c1] = xb[i_k_stage_c1] + __hcse_6 * __cse_0
    end
    return nothing
end

function inl_argexpr(loss, w, x, t, i_n)
    __inl_stage_1_4 = i_n - 1
    for i_k_stage_c1 = 1:__inl_stage_1_4
        t[i_k_stage_c1] = t[i_k_stage_c1] + w[i_k_stage_c1] * x[i_k_stage_c1]
    end
    for i_j = 1:i_n
        loss[1] = loss[1] + t[i_j] ^ 2
    end
end
