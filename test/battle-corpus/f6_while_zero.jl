# f6_while_zero
# While that runs zero times for most draws.
# LOOP i1 I K
# WHILE r1 T
function f6_while_zero(x, y, n)
    for i1 = 1:n
        r1 = x[i1] - 100.0
        while r1 > 0.0
            r1 = r1 - 1.0
        end
        y[i1] = r1
    end
    return nothing
end
