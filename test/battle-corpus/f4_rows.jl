# f4_rows
# Rows independent, recurrence along each row.
# LOOP r1 I K
# LOOP c1 S T
function f4_rows(x, u, n, m)
    for r1 = 1:n
        for c1 = 2:m
            u[r1, c1] = 0.5 * u[r1, c1 - 1] + x[r1, c1]
        end
    end
    return nothing
end
