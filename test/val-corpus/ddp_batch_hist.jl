# ddp_batch_hist(loss, u, a, hist, v, i_n, i_k)
#
# The shape that needs the `reduced` keyword. Array hist accumulates at
# an index derived from the data, so the index varies with the enclosing
# loop. That is the same write shape as a per-sample output such as
# matvec_loss's v, and the kernel does not say which reading applies.
#
# Here the histogram is meant to span the whole batch: every sample adds
# to the same bins, so the ranks hold partial sums that must be summed.
# Declaring reduced = [:hist] says so. Naming it in per_sample instead
# would be accepted and would give each rank its own partial histogram,
# which is why the refusal for an unresolved array has to be loud.
#
# loss: length-1 output array, accumulated in place
# u: input features for one sample, length i_n
# a: elementwise weight, length i_n
# hist: batch histogram, length i_k, caller-zeroed, accumulated in place
# v: scratch activations, length i_n
# i_n: feature count
# i_k: bin count
function ddp_batch_hist(loss, u, a, hist, v, i_n, i_k)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        i_bin = mod(i_x2 - 1, i_k) + 1
        hist[i_bin] = hist[i_bin] + v[i_x2] * v[i_x2]
    end
    for i_x3 = 1:i_n
        loss[1] = loss[1] + v[i_x3] ^ 2
    end
    return nothing
end
