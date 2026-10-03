# red_refuse_barevar(loss, u, i_n)
#
# Negative subject. The loop variable appears outside an index position,
# so each term depends on the iteration number itself. A mapreduce over
# values alone cannot see the index, and cgen_bare_var_outside_index must
# refuse.
#
# loss: length-1 output array, accumulated in place
# u: vector of length i_n
# i_n: length
function red_refuse_barevar(loss, u, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] * i_x
    end
    return nothing
end
