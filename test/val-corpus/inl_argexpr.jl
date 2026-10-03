# inl_argexpr(loss, w, x, t, i_n)
#
# An expression passed for a read-only scalar parameter. The inliner binds
# it to a temporary at the call site, so the substitution stays
# symbol-to-symbol and the expression is evaluated once per call rather
# than once per use.
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
function inl_argexpr(loss, w, x, t, i_n)
    stage(t, w, x, i_n - 1)
    for i_j = 1:i_n
        loss[1] = loss[1] + t[i_j] ^ 2
    end
    return nothing
end
