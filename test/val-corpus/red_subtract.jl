# red_subtract(loss, u, v, i_n)
#
# The `-` operator form: the accumulator is reduced by the term rather
# than increased. cgen_idiomatic_scalar_reduction accepts `target - term`
# but must refuse `term - target`, which is not a reduction at all.
#
# loss: length-1 output array, accumulated in place
# u, v: vectors of length i_n
# i_n: length
function red_subtract(loss, u, v, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] - u[i_x] * v[i_x]
    end
    return nothing
end
