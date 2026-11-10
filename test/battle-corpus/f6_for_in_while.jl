# f6_for_in_while
# Sequential for inside a while inside an independent loop.
# LOOP i1 I K
# WHILE r1 T
# LOOP c1 I T
function f6_for_in_while(a, y, tol, n, m)
    for i1 = 1:n
        r1 = 1.0
        s1 = 0.0
        while r1 > tol
            for c1 = 1:m
                s1 = s1 + a[i1, c1] * r1
            end
            r1 = 0.5 * r1
        end
        y[i1] = s1
    end
    return nothing
end
