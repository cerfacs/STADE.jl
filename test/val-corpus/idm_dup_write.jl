# idm_dup_write(loss, mask, i_idx, c, i_n)
#
# Counter-example for the plain-write contract. Several iterations write
# the same value to one element of mask, through a gathered index. The
# forward result does not depend on which iteration writes last. The
# adjoint consumes the shadow of mask at the same gathered index. Run in
# parallel, two iterations read the shadow before either zeroes it, so the
# gradient of c counts one element more than once. The ownership check
# must report this kernel. The gate accepts it, on the contract that
# iterations own distinct elements.
#
# loss: length-1 output array, accumulated in place
# mask: output array of length i_n
# i_idx: gather index array of length i_n, values in 1:i_n
# c: scalar value written to every selected element
# i_n: number of iterations
function idm_dup_write(loss, mask, i_idx, c, i_n)
    for i_t = 1:i_n
        i_j = i_idx[i_t]
        mask[i_j] = c * 2.0
    end
    for i_k = 1:i_n
        loss[1] = loss[1] + mask[i_k] * mask[i_k]
    end
    return nothing
end
