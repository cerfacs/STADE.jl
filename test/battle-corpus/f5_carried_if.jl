# f5_carried_if
# Branch decides whether the dependence exists (S).
# LOOP i1 S H
function f5_carried_if(x, u, n)
    for i1 = 2:n
        if x[i1] > 0.0
            u[i1] = u[i1 - 1] + x[i1]
        else
            u[i1] = u[i1 - 1]
        end
    end
    return nothing
end
