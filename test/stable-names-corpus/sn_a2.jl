# sn_a2
# An inner while and an earlier top-level while with the same text at the same position in their bodies.
# The outer while contains the inner one, so its tag differs from both.
function sn_a2(x, y, n, m)
    k = 1
    while k <= n
        y[k] = y[k] + x[k] * x[k]
        k = k + 1
    end
    j = 1
    while j <= m
        k = 1
        while k <= n
            y[k] = y[k] + x[k] * x[k]
            k = k + 1
        end
        j = j + 1
    end
    return nothing
end
