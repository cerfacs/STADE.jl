# float_trip(x, lim, out)
#
# A while loop whose trip count depends on the value of x. The count is piecewise constant in x.
#
# x: length-1 input array
# lim: threshold
# out: length-1 output array; out[1] receives the result
function float_trip(x, lim, out)
    s = x[1]
    k = 1
    while s < lim
        s = s * 2.0
        k += 1
    end
    out[1] = s * s + k * x[1]
    return nothing
end
