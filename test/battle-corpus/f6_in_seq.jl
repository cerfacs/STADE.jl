# f6_in_seq
# While inside a sequential loop.
# LOOP t1 S H
# WHILE r1 H
function f6_in_seq(x, u, tol, nt)
    for t1 = 2:nt
        r1 = x[t1]
        while r1 > tol
            r1 = 0.5 * r1
        end
        u[t1] = u[t1 - 1] + r1
    end
    return nothing
end
