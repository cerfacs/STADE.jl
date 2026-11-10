# f5_boundary
# Branch on the loop index (boundary special case).
# LOOP i1 I K
function f5_boundary(x, y, n)
    for i1 = 1:n
        if i1 == 1
            y[i1] = x[i1]
        else
            y[i1] = x[i1] - x[i1 - 1]
        end
    end
    return nothing
end
