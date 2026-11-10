# f9_gather_write
# Plain write through a gathered index: disjointness cannot be proven.
# LOOP i1 S H
function f9_gather_write(x, y, idx, n)
    for i1 = 1:n
        j1 = idx[i1]
        y[j1] = x[i1]
    end
    return nothing
end
