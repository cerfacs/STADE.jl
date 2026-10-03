# ddp_written_per_sample(loss, u, a, v, i_n)
#
# Negative subject for bgen_check_declaration. Array v is scratch: the
# kernel writes it by plain overwrite. Naming v in per_sample must be
# refused, because per_sample classifies read-only arrays only, and a
# written array already has a derived role.
#
# The kernel itself is ordinary and passes every corpus oracle. Only the
# declaration is wrong, which is the point: the refusal must come from
# the declaration check and not from the kernel being malformed.
#
# loss: length-1 output array, accumulated in place
# u: input features for one sample, length i_n
# a: elementwise weight, length i_n
# v: scratch activations, length i_n
# i_n: feature count
function ddp_written_per_sample(loss, u, a, v, i_n)
    for i_x = 1:i_n
        v[i_x] = a[i_x] * u[i_x]
    end
    for i_x2 = 1:i_n
        loss[1] = loss[1] + v[i_x2] ^ 2
    end
    return nothing
end
