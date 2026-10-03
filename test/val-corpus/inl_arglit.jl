# inl_arglit(loss, w, x, t, i_n)
#
# A literal passed for a read-only scalar parameter. Before v0.4.1 both
# this and inl_argexpr were refused, and the caller had to hoist a variable
# by hand.
#
# loss: length-1 output array, accumulated in place
# w, x: inputs of length i_n
# t: scratch, length i_n
# i_n: length
function stage(t, w, x, i_m)
    for i_k = 1:i_m
        t[i_k] = t[i_k] + w[i_k] * x[i_k]
    end
    return nothing
end
function inl_arglit(loss, w, x, t, i_n)
    stage(t, w, x, 3)
    for i_j = 1:i_n
        loss[1] = loss[1] + t[i_j] ^ 2
    end
    return nothing
end
