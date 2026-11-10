# f1_two
# Two independent loops in sequence.
# LOOP i1 I K
# LOOP i2 I K
function f1_two(x, y, z, n, m)
    for i1 = 1:n
        y[i1] = x[i1] + 1.0
    end
    for i2 = 1:m
        z[i2] = x[i2] * 3.0
    end
    return nothing
end
