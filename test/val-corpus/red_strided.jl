# red_strided(loss, u, v, i_n)
#
# A step of 2. The view is strided, so it is not contiguous device
# memory, and the reduction must still read exactly the visited elements.
# A whole-array lowering would be wrong by every odd-indexed term.
#
# loss: length-1 output array, accumulated in place
# u, v: vectors of length i_n
# i_n: length
function red_strided(loss, u, v, i_n)
    for i_x = 1:2:i_n
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
    return nothing
end
