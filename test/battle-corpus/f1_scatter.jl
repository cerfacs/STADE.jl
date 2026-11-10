# f1_scatter
# Scatter accumulate: commutative, so independent.
# LOOP i1 I K
function f1_scatter(x, v, idx, n)
    for i1 = 1:n
        j1 = idx[i1]
        v[j1] = v[j1] + x[i1]
    end
    return nothing
end
