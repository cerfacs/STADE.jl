# f6_two_while
# Two sequential whiles at host level.
# WHILE k1 H
# WHILE k2 H
function f6_two_while(x, y, n)
    k1 = 1
    while k1 <= n
        y[k1] = x[k1]
        k1 = k1 + 1
    end
    k2 = 1
    while k2 <= n
        y[k2] = y[k2] * 2.0
        k2 = k2 + 1
    end
    return nothing
end
