# f5_inner_loop_branch
# Inner sequential loop under a branch in an independent loop.
# LOOP r1 I K
# LOOP c1 S T
function f5_inner_loop_branch(x, u, n, m)
    for r1 = 1:n
        if x[r1, 1] > 0.0
            for c1 = 2:m
                u[r1, c1] = 0.5 * u[r1, c1 - 1] + x[r1, c1]
            end
        else
            u[r1, 1] = 0.0
        end
    end
    return nothing
end
