# f2_two_seq
# Two sequential loops in sequence.
# LOOP i1 S H
# LOOP i2 S H
function f2_two_seq(u, v, n)
    for i1 = 2:n
        u[i1] = 0.5 * u[i1 - 1] + 1.0
    end
    for i2 = 2:n
        v[i2] = v[i2 - 1] + u[i2]
    end
    return nothing
end
