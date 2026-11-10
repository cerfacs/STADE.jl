# f2_branch_carry
# Branch with a carried dependence in one arm.
# LOOP i1 S H
function f2_branch_carry(x, u, n)
    for i1 = 2:n
        if x[i1] > 0.0
            u[i1] = u[i1 - 1] + 1.0
        else
            u[i1] = x[i1]
        end
    end
    return nothing
end
