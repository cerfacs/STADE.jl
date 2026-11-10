# f4_siblings
# Independent outer loop with an S and an I loop as siblings.
# LOOP r1 I K
# LOOP c1 S T
# LOOP c2 I T
function f4_siblings(x, u, w, n, m)
    for r1 = 1:n
        for c1 = 2:m
            u[r1, c1] = 0.5 * u[r1, c1 - 1] + x[r1, c1]
        end
        for c2 = 1:m
            w[r1, c2] = u[r1, c2] * 2.0
        end
    end
    return nothing
end
