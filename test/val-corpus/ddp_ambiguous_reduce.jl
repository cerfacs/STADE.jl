# ddp_ambiguous_reduce(loss, u, a, v, i_n)
#
# Negative subject for the ambiguous reduction shape. Array v takes only
# additive writes, because the caller sets it to zero before the call.
# That makes it structurally identical to a :reduced output such as loss,
# yet it holds a per-sample result and must not be summed across ranks.
#
# No analysis of the kernel separates the two. bgen_reduced_outputs must
# therefore refuse this shape and name the array, rather than guess.
# matvec_loss carries the same shape and serves as a second witness.
#
# loss: length-1 output array, accumulated in place
# u: input features for one sample, length i_n
# a: elementwise weight, length i_n
# v: per-sample output, caller-zeroed, accumulated in place
# i_n: feature count
function ddp_ambiguous_reduce(loss, u, a, v, i_n)
    for i_x = 1:i_n
        v[i_x] = v[i_x] + a[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    return nothing
end
