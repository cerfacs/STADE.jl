function initstacks_idm_dup_write_b()
    return nothing
end

function idm_dup_write_b(loss, lossb, mask, maskb, i_idx, c, cb, i_n)
    for i_t = 1:i_n
        i_j = i_idx[i_t]
        mask[i_j] = c * 2.0
    end
    for i_k = 1:i_n
        __cse_0 = mask[i_k]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for i_k = i_n:-1:1
        __cse_1 = mask[i_k] * lossb[1]
        maskb[i_k] = maskb[i_k] + __cse_1
        maskb[i_k] = maskb[i_k] + __cse_1
    end
    for i_t = i_n:-1:1
        i_j = i_idx[i_t]
        __oldb_0 = maskb[i_j]
        maskb[i_j] = 0.0
        cb = cb + 2.0__oldb_0
    end
    return cb
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
