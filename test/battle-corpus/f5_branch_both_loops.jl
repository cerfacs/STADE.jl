# f5_branch_both_loops
# Both arms hold an inner loop under an independent outer loop.
# LOOP r1 I K
# LOOP c1 I T
# LOOP c2 I T
function f5_branch_both_loops(x, y, z, n, m)
    for r1 = 1:n
        if x[r1, 1] > 0.0
            for c1 = 1:m
                y[r1, c1] = x[r1, c1]
            end
        else
            for c2 = 1:m
                z[r1, c2] = x[r1, c2]
            end
        end
    end
    return nothing
end
