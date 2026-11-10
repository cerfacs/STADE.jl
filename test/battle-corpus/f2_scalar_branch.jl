# f2_scalar_branch
# Active scalar carried through a branch.
# LOOP i1 S H
function f2_scalar_branch(x, y, n)
    s1 = 0.0
    for i1 = 1:n
        if x[i1] > 0.0
            s1 = 0.5 * s1 + x[i1]
        else
            s1 = 0.0
        end
        y[i1] = s1
    end
    return nothing
end
