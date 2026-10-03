# red_single_cube(loss, u, i_n)
#
# One array, and a term that is NOT `u[i]^2`, so it misses the
# sum(abs2, .) special case and lowers to a single-collection mapreduce
# under an anonymous closure. That is the one arity Julia cannot give a
# neutral element to, so without an explicit init this kernel raises
# "reducing over an empty collection is not allowed" whenever i_n is 0.
#
# loss: length-1 output array, accumulated in place
# u: vector of length i_n
# i_n: length, and the shape that must be exercised at 0
function red_single_cube(loss, u, i_n)
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 3
    end
    return nothing
end
