# red_reverse(loss, u, v, i_n)
#
# A descending loop. The emitted view is `i_n:-1:1`, reversed, and a sum
# over it must equal the ascending one. Reverse sweeps produce this shape
# routinely, so it is not exotic.
#
# loss: length-1 output array, accumulated in place
# u, v: vectors of length i_n
# i_n: length
function red_reverse(loss, u, v, i_n)
    for i_x = i_n:-1:1
        loss[1] = loss[1] + u[i_x] * v[i_x]
    end
    return nothing
end
