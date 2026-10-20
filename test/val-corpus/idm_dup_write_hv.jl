function initstacks_idm_dup_write_b()
    return nothing
end

function idm_dup_write_hv(loss, lossb, mask, maskb, i_idx, c, cb, i_n, lossd, lossbd, maskd, maskbd, cd, cbd)
    for i_t = 1:i_n
        i_j = i_idx[i_t]
        maskd[i_j] = 2.0cd
        mask[i_j] = c * 2.0
    end
    for i_k = 1:i_n
        __cse_0d = maskd[i_k]
        __cse_0 = mask[i_k]
        __hcse_0 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_0 + __hcse_0)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for i_k = i_n:-1:1
        __hcse_1 = lossb[1]
        __hcse_2 = mask[i_k]
        __cse_1d = __hcse_1 * maskd[i_k] + __hcse_2 * lossbd[1]
        __cse_1 = __hcse_2 * __hcse_1
        maskbd[i_k] = maskbd[i_k] + __cse_1d
        maskb[i_k] = maskb[i_k] + __cse_1
        maskbd[i_k] = maskbd[i_k] + __cse_1d
        maskb[i_k] = maskb[i_k] + __cse_1
    end
    for i_t = i_n:-1:1
        i_j = i_idx[i_t]
        __oldb_0d = maskbd[i_j]
        __oldb_0 = maskb[i_j]
        maskbd[i_j] = 0.0
        maskb[i_j] = 0.0
        cbd = cbd + 2.0__oldb_0d
        cb = cb + 2.0__oldb_0
    end
    return (cb, cbd)
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
