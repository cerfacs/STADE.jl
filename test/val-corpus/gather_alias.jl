# gather_alias(y, x, i_idx, i_n)
#
# Regression kernel for a wrong adjoint STADE produced silently until
# agen_backward_assign started capturing the incoming adjoint before
# consuming the output slot.
#
# The loop is sequential (rule 1 permits that, and says to say so): iteration
# i_r reads y at a gathered index an earlier iteration may already have
# written, so the iterations only give the right answer in order.
#
# The defect was not the gather. It was that `y[i_r]` and `y[i_k]` are
# different expressions, so the syntactic self-reference match in
# agen_distribute! never fired, and the backward sweep distributed into
# `yb[i_k]` and then zeroed `yb[i_r]` -- discarding the contribution it had
# just accumulated on every iteration where i_idx[i_r] == i_r. The gradient
# was wrong in proportion to how often the index aliased: exact at zero
# fixed points, off by ~100% at the identity.
#
# The baseline generator draws i_idx randomly over 1:i_n, so fixed points
# appear on essentially every draw and this kernel exercises the shape
# without needing a rigged index. Only the adjoint and dot-product oracles
# see it; the tangent and HVP both passed throughout.
#
# y: state array of length i_n, updated in place
# x: input array of length i_n
# i_idx: gather index array of length i_n, values in 1:i_n
# i_n: array length
function gather_alias(y, x, i_idx, i_n)
    for i_r = 1:i_n
        i_k = i_idx[i_r]
        # reads y at a gathered index, writes y at the loop index
        y[i_r] = y[i_k] * x[i_r]
    end
    return nothing
end
