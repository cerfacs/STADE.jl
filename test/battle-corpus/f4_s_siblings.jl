# f4_s_siblings
# Sequential outer loop with an S and an I loop as siblings.
# LOOP t1 S H
# LOOP c1 S H
# LOOP c2 I K
function f4_s_siblings(x, u, w, nt, m)
    for t1 = 2:nt
        for c1 = 2:m
            u[t1, c1] = 0.5 * u[t1, c1 - 1] + x[t1, c1]
        end
        for c2 = 1:m
            w[t1, c2] = u[t1, c2] * 0.5 + w[t1 - 1, c2]
        end
    end
    return nothing
end
