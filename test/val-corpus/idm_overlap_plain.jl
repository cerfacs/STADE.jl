# idm_overlap_plain(a, x, y, loss, i_n)
#
# Must stay on the host. Iteration i_t writes a[i_t] and a[i_t + 1]. Iteration i_t + 1 writes a[i_t + 1] again, and in sequence
# its write is the one that survives. Run as parallel threads, either write can win. Both indices are affine in i_t with the same
# stride, so the overlap is provable and the gate refuses the loop. The offload report must show it refused for `a`.
#
# a: output array of length i_n + 1
# x, y: input arrays of length i_n
# loss: length-1 output array, accumulated in place
function idm_overlap_plain(a, x, y, loss, i_n)
    for i_t = 1:i_n
        a[i_t] = x[i_t] * 2.0
        a[i_t + 1] = y[i_t] * 3.0
    end
    for i_k = 1:i_n
        loss[1] = loss[1] + a[i_k] * a[i_k]
    end
    return nothing
end
