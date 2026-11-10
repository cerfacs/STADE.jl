# f2_counter
# Integer counter carried between iterations.
# LOOP i1 S H
function f2_counter(x, y, n)
    k1 = 0
    for i1 = 1:n
        k1 = k1 + 1
        y[k1] = x[i1]
    end
    return nothing
end
