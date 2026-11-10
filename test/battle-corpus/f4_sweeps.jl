# f4_sweeps
# Outer sweep loop S, inner Jacobi update and copy I.
# LOOP t1 S H
# LOOP i1 I K
# LOOP i2 I K
function f4_sweeps(u, v, nt, n)
    for t1 = 1:nt
        for i1 = 2:n - 1
            v[i1] = 0.5 * (u[i1 - 1] + u[i1 + 1])
        end
        for i2 = 2:n - 1
            u[i2] = v[i2]
        end
    end
    return nothing
end
