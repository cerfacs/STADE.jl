# f4_triangular
# Independent outer loop, triangular inner loop.
# LOOP r1 I K
# LOOP c1 I T
function f4_triangular(x, y, n)
    for r1 = 1:n
        s1 = 0.0
        for c1 = 1:r1
            s1 = s1 + x[c1]
        end
        y[r1] = s1
    end
    return nothing
end
