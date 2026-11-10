# f9_halo
# Two plain writes at i1 and i1 + 1 overlap across iterations.
# LOOP i1 S H
function f9_halo(x, y, n)
    for i1 = 1:n
        y[i1] = x[i1]
        y[i1 + 1] = 2.0 * x[i1]
    end
    return nothing
end
