# f9_scatter_rw
# Scatter accumulate into an array that another statement of the loop reads.
# LOOP i1 S H
function f9_scatter_rw(x, v, idx, y, n)
    for i1 = 1:n
        j1 = idx[i1]
        v[j1] = v[j1] + x[i1]
        y[i1] = v[j1]
    end
    return nothing
end
