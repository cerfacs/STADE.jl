function initstacks_inl_arglit_b()
    return nothing
end

function inl_arglit_b(loss, lossb, w, wb, x, xb, t, tb, i_n)
    __inl_stage_1_4 = 3
    for i_k_stage_c1 = 1:__inl_stage_1_4
        t[i_k_stage_c1] = t[i_k_stage_c1] + w[i_k_stage_c1] * x[i_k_stage_c1]
    end
    for i_j = 1:i_n
        loss[1] = loss[1] + t[i_j] ^ 2
    end
    __inl_stage_1_4 = 3
    for i_j = i_n:-1:1
        tb[i_j] = tb[i_j] + (2 * t[i_j]) * lossb[1]
    end
    for i_k_stage_c1 = __inl_stage_1_4:-1:1
        __cse_0 = tb[i_k_stage_c1]
        wb[i_k_stage_c1] = wb[i_k_stage_c1] + x[i_k_stage_c1] * __cse_0
        xb[i_k_stage_c1] = xb[i_k_stage_c1] + w[i_k_stage_c1] * __cse_0
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
