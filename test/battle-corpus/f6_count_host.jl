# f6_count_host
# Counting while at host level: not an independent loop.
# WHILE k1 H
function f6_count_host(x, y, n)
    k1 = 1
    while k1 <= n
        y[k1] = x[k1] * 2.0
        k1 = k1 + 1
    end
    return nothing
end
