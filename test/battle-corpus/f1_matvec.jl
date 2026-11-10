# f1_matvec
# Row loop independent, inner dot is a scalar reduction.
# LOOP i1 I K
# LOOP i2 I T
function f1_matvec(a, x, y, n, m)
    for i1 = 1:n
        s1 = 0.0
        for i2 = 1:m
            s1 = s1 + a[i1, i2] * x[i2]
        end
        y[i1] = s1
    end
    return nothing
end
