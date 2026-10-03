# ddp_affine(loss, u, a, b, v, i_n)
#
# Minimal subject for the bgen_ stage. One affine layer followed by a
# squared-error loss. Every bgen_ role appears exactly once, which is
# what makes this the kernel to read first when a role table looks wrong.
#
# With per_sample = [:u], bgen_ derives:
#   a, b   :shared   read-only, not named per_sample
#   loss   :reduced  every write adds to loss itself
#   v      :scratch  written by plain overwrite
#   u      :per_sample
#
# loss: length-1 output array, accumulated in place
# u: input features for one sample, length i_n
# a: elementwise weight, length i_n
# b: elementwise bias, length i_n
# v: scratch activations, length i_n
# i_n: feature count
function ddp_affine(loss, u, a, b, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x] + b[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    return nothing
end
