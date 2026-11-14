# pv_a20(xa, xb, xc, y, u, z, v, n, m1, m2)
#
# The phases of pv_a13 with a bound of its own for each accumulating loop, so a draw can empty one phase and keep the others. After the reset, a zero-trip accumulating loop leaves the box at its start value and the reader must see that value.
#
# xa: input of length m1
# xb: input of length m2
# xc: input of length n
# y: output of length m1
# u: output of length m2
# z: output of length n
# v: output of length n
# n: number of points the readers cover
# m1: trip count of the first accumulating loop
# m2: trip count of the second accumulating loop
function pv_a20(xa, xb, xc, y, u, z, v, n, m1, m2)
    acc = 0.0
    for i = 1:m1
        y[i] = xa[i] * xa[i]
        acc += y[i]
    end
    for i = 1:n
        z[i] = xc[i] * acc
    end
    acc = 0.0
    for i = 1:m2
        u[i] = 2.0 * xb[i] + 1.0
        acc += u[i]
    end
    for i = 1:n
        v[i] = xc[i] * acc
    end
    return nothing
end
