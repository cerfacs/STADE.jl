# pv_a19(cnt, x, y, z, n)
#
# An Int scalar ARGUMENT is counted in one loop and used as the bound of the next: the box appears in the host's grid expression and in the kernel's guard.
#
# cnt: Int counter argument, its value after the first loop is the bound of the second
# x: input of length n
# y: output of length n + cnt
# z: output of length n + cnt
# n: number of points
function pv_a19(cnt, x, y, z, n)
    c0 = cnt
    for i = 1:n
        y[i] = 2.0 * x[i]
        cnt += 1
    end
    for i = 1:cnt - c0
        z[i] = y[i] + 1.0
    end
    return nothing
end
