# f4_ind_then_seq
# Independent loop followed by sequential loop.
# LOOP i1 I K
# LOOP i2 S H
function f4_ind_then_seq(x, u, n)
    for i1 = 1:n
        u[i1] = x[i1] * 2.0
    end
    for i2 = 2:n
        u[i2] = 0.5 * u[i2 - 1] + u[i2]
    end
    return nothing
end
