# sn_a0
# Two identical whiles in one body. Their counters differ only by position.
function sn_a0(x, y, n)
    k = 1
    while k <= n
        y[k] = y[k] + x[k] * x[k]
        k = k + 1
    end
    k = 1
    while k <= n
        y[k] = y[k] + x[k] * x[k]
        k = k + 1
    end
    return nothing
end
