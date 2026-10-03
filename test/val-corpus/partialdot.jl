# partialdot(loss, u, v, i_n)
#
# A scalar reduction whose loop does NOT start at 1. Every other
# reduction in this corpus runs 1:n, so nothing here distinguished "sum
# the elements this loop visited" from "sum the whole array", and the
# GPU reduction path emitted the second. On a Tesla V100 at length 8 the
# loop gives 70.0 and the whole-array form 72.0.
#
# keep_all_atomic = false is the flag that reaches that path. Generate
# with it when using this kernel as a witness.
#
# loss: length-1 output array, accumulated in place
# u, v: vectors of length i_n
# i_n: length
function partialdot(loss, u, v, i_n)
    for i_x = 2:i_n
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
    return nothing
end
