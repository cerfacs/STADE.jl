function inl_arglit_d(loss, lossd, w, wd, x, xd, t, td, i_n)
    __inl_stage_1_4d = 0.0
    __inl_stage_1_4 = 3
    for i_k_stage_c1 = 1:__inl_stage_1_4
        __cse_0 = x[i_k_stage_c1]
        __cse_1 = w[i_k_stage_c1]
        td[i_k_stage_c1] = td[i_k_stage_c1] + (__cse_0 * wd[i_k_stage_c1] + __cse_1 * xd[i_k_stage_c1])
        t[i_k_stage_c1] = t[i_k_stage_c1] + __cse_1 * __cse_0
    end
    for i_j = 1:i_n
        __cse_2 = t[i_j]
        lossd[1] = lossd[1] + (2__cse_2) * td[i_j]
        loss[1] = loss[1] + __cse_2 ^ 2
    end
    return nothing
end

function inl_arglit(loss, w, x, t, i_n)
    __inl_stage_1_4 = 3
    for i_k_stage_c1 = 1:__inl_stage_1_4
        t[i_k_stage_c1] = t[i_k_stage_c1] + w[i_k_stage_c1] * x[i_k_stage_c1]
    end
    for i_j = 1:i_n
        loss[1] = loss[1] + t[i_j] ^ 2
    end
end
