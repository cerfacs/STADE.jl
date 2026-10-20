function idm_dup_write_d(loss, lossd, mask, maskd, i_idx, c, cd, i_n)
    for i_t = 1:i_n
        i_jd = 0.0
        i_j = i_idx[i_t]
        maskd[i_j] = 2.0cd
        mask[i_j] = c * 2.0
    end
    for i_k = 1:i_n
        __cse_0 = mask[i_k]
        __cse_1 = __cse_0 * maskd[i_k]
        lossd[1] = lossd[1] + (__cse_1 + __cse_1)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    return nothing
end

function idm_dup_write(loss, mask, i_idx, c, i_n)
    for i_t = 1:i_n
        i_j = i_idx[i_t]
        mask[i_j] = c * 2.0
    end
    for i_k = 1:i_n
        loss[1] = loss[1] + mask[i_k] * mask[i_k]
    end
end
