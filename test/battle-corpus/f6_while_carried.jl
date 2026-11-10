# f6_while_carried
# Host while whose body is a carried recurrence.
# WHILE k1 H
# LOOP i1 S H
function f6_while_carried(u, nit, n)
    k1 = 1
    while k1 <= nit
        for i1 = 2:n
            u[i1] = 0.5 * u[i1 - 1] + u[i1]
        end
        k1 = k1 + 1
    end
    return nothing
end
