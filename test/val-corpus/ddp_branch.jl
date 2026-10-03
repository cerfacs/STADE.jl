# ddp_branch(loss, u, a, b, v, i_n)
#
# Per-sample subject with an :if on per-sample data. The branch condition
# reads u, so two samples on one rank can take two different arms. That
# makes a :branch snapshot site, and the site is written on every local
# call rather than once per process.
#
# This is the shape that breaks if bgen_reset_local! misses a buffer:
# sample 2 would take its own arm while reading sample 1's adjoint.
#
# loss: length-1 output array, accumulated in place
# u: input features for one sample, length i_n
# a: elementwise weight, length i_n
# b: elementwise bias, length i_n
# v: scratch activations, length i_n
# i_n: feature count
function ddp_branch(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        if u[i_x] > 0.0
            v[i_x] = a[i_x] * u[i_x]
        else
            v[i_x] = b[i_x] * u[i_x] * u[i_x]
        end
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    return nothing
end
