# f2_gs
# In-place Gauss-Seidel sweep.
# LOOP i1 S H
function f2_gs(u, n)
    for i1 = 2:n - 1
        u[i1] = 0.5 * (u[i1 - 1] + u[i1 + 1])
    end
    return nothing
end
