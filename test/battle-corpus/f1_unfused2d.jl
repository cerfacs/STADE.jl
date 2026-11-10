# f1_unfused2d
# Unfused rectangular nest: outer K, inner T.
# LOOP i1 I K
# LOOP i2 I T
function f1_unfused2d(x, y, n, m)
    for i1 = 1:n
        for i2 = 1:m
            y[i1, i2] = x[i1, i2] * 2.0
        end
    end
    return nothing
end
