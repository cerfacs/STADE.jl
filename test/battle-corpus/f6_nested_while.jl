# f6_nested_while
# Two nested whiles inside an independent loop.
# LOOP i1 I K
# WHILE r1 T
# WHILE q1 T
function f6_nested_while(x, y, tol, n)
    for i1 = 1:n
        r1 = x[i1]
        s1 = 0.0
        while r1 > tol
            q1 = r1
            while q1 > tol
                q1 = 0.5 * q1
                s1 = s1 + 1.0
            end
            r1 = 0.5 * r1
        end
        y[i1] = s1
    end
    return nothing
end
