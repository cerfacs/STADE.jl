# red_refuse_invariant(loss, u, c, i_n)
#
# Negative subject. The term reads `c[1]`, an element NOT indexed by the
# loop variable, so the reduction cannot be expressed as a mapreduce over
# aligned views. cgen_idiomatic_scalar_reduction must refuse it and leave
# the loop to the atomic kernel.
#
# loss: length-1 output array, accumulated in place
# u: vector of length i_n
# c: length-1 array, a loop-invariant factor
# i_n: length
function red_refuse_invariant(loss, u, c, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * c[1]
    end
    return nothing
end
