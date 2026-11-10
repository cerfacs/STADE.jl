# f1_scale
# One independent loop, one write per index.
# LOOP i1 I K
function f1_scale(x, y, a, n)
    for i1 = 1:n
        y[i1] = a * x[i1]
    end
    return nothing
end
