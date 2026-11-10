# f9_stride2
# Two plain writes at 2*i1 and 2*i1 - 1 cover disjoint sets.
# LOOP i1 I K
function f9_stride2(x, y, n)
    for i1 = 1:n
        y[2 * i1] = x[i1]
        y[2 * i1 - 1] = 2.0 * x[i1]
    end
    return nothing
end
