# f4_stack3
# Independent, sequential, independent loops in sequence.
# LOOP i1 I K
# LOOP i2 S H
# LOOP i3 I K
function f4_stack3(x, u, y, n)
    for i1 = 1:n
        u[i1] = x[i1] * 2.0
    end
    for i2 = 2:n
        u[i2] = 0.5 * u[i2 - 1] + u[i2]
    end
    for i3 = 1:n
        y[i3] = u[i3] + 1.0
    end
    return nothing
end
