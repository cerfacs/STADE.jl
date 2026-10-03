# red_three_arrays(loss, u, v, w, i_n)
#
# Three arrays in one reduction term. The emitted closure takes three
# arguments and reduces three views at once, which no other corpus
# kernel does.
#
# loss: length-1 output array, accumulated in place
# u, v, w: vectors of length i_n
# i_n: length
function red_three_arrays(loss, u, v, w, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + (u[i_x] * v[i_x]) * w[i_x]
    end
    return nothing
end
