# f4_adjacent
# Two adjacent independent loops with equal bounds.
# LOOP i1 I K
# LOOP i2 I K
function f4_adjacent(x, y, z, n)
    for i1 = 1:n
        y[i1] = x[i1] * 2.0
    end
    for i2 = 1:n
        z[i2] = x[i2] + 1.0
    end
    return nothing
end
