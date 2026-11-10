# f6_solver
# Per-element solver: independent outer loop, while inside.
# LOOP i1 I K
# WHILE r1 T
function f6_solver(x, y, tol, n)
    for i1 = 1:n
        r1 = x[i1]
        while r1 > tol
            r1 = 0.5 * r1
        end
        y[i1] = r1
    end
    return nothing
end
