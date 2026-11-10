# f4_time2d
# Time loop S around an independent space loop.
# LOOP t1 S H
# LOOP i1 I K
function f4_time2d(x, u, nt, n)
    for t1 = 2:nt
        for i1 = 1:n
            u[t1, i1] = 0.5 * u[t1 - 1, i1] + x[i1]
        end
    end
    return nothing
end
