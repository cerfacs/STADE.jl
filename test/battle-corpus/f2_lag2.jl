# f2_lag2
# Two-lag recurrence.
# LOOP i1 S H
function f2_lag2(u, n)
    for i1 = 3:n
        u[i1] = u[i1 - 1] + u[i1 - 2]
    end
    return nothing
end
