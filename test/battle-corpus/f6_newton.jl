# f6_newton
# Host while (convergence) around an independent loop.
# WHILE e1 H
# LOOP i1 I K
# LOOP i2 I K
function f6_newton(x, y, err, tol, n)
    e1 = 1.0
    while e1 > tol
        for i1 = 1:n
            y[i1] = 0.5 * (y[i1] + x[i1] / y[i1])
        end
        e1 = 0.0
        for i2 = 1:n
            err[1] = err[1] + y[i2]
        end
        e1 = e1 * 0.5
    end
    return nothing
end
