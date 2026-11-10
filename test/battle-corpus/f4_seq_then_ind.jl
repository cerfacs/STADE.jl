# f4_seq_then_ind
# Sequential loop followed by independent loop.
# LOOP i1 S H
# LOOP i2 I K
function f4_seq_then_ind(u, y, n)
    for i1 = 2:n
        u[i1] = 0.5 * u[i1 - 1] + 1.0
    end
    for i2 = 1:n
        y[i2] = u[i2] * 2.0
    end
    return nothing
end
