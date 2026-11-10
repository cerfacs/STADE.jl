# f1_flat2d
# Pre-fused rectangular nest (skill rule 2).
# LOOP fz I K
function f1_flat2d(x, y, n, m)
    for fz = 1:n * m
        r1 = div(fz - 1, m) + 1
        c1 = mod(fz - 1, m) + 1
        y[r1, c1] = x[r1, c1] * 2.0
    end
    return nothing
end
