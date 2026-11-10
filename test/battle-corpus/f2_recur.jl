# f2_recur
# First-order recurrence.
# LOOP i1 S H
function f2_recur(u, n)
    for i1 = 2:n
        u[i1] = 0.5 * u[i1 - 1] + 1.0
    end
    return nothing
end
