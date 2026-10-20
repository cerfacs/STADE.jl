# if_top_two(x, y, i_n)
#
# Two `if`s outside any loop, each giving a scalar (`s`, `t`) its value after a plain initialisation; a later loop reads both. See
# if_top_scalar.
#
# x: input, length >= max(3, i_n)
# y: output accumulator, length i_n, updated in place
# i_n: loop length
function if_top_two(x, y, i_n)
    s = 0.0
    if i_n > 2
        s = x[1] * x[2]
    end
    t = 0.0
    if i_n > 3
        t = x[2] * x[3]
    end
    for i = 1:i_n
        y[i] = y[i] + s * t * x[i]
    end
    return nothing
end
