# f9_gather_rmw
# Read-modify-write through a gathered index.
# LOOP i1 S H
function f9_gather_rmw(x, y, idx, n)
    for i1 = 1:n
        j1 = idx[i1]
        y[j1] = y[j1] * x[i1]
    end
    return nothing
end
