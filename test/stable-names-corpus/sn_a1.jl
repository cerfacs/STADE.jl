# sn_a1
# Identical whiles in the two arms of one if. They share a tag, and only one arm runs.
function sn_a1(x, y, n)
    if n > 4
        k = 1
        while k <= n
            y[k] = y[k] + x[k] * x[k]
            k = k + 1
        end
    else
        k = 1
        while k <= n
            y[k] = y[k] + x[k] * x[k]
            k = k + 1
        end
    end
    return nothing
end
