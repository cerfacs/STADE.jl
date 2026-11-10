# f5_branch_in_seq
# Branch with an independent loop inside a sequential loop.
# LOOP t1 S H
# LOOP i1 I K
function f5_branch_in_seq(x, u, nt, n)
    for t1 = 2:nt
        if x[t1] > 0.0
            for i1 = 1:n
                u[t1, i1] = 0.5 * u[t1 - 1, i1] + x[t1]
            end
        end
    end
    return nothing
end
