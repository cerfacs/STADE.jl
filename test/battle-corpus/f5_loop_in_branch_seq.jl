# f5_loop_in_branch_seq
# Independent loop in one arm, sequential loop in the other.
# LOOP i1 I K
# LOOP i2 S H
function f5_loop_in_branch_seq(c, x, u, y, n)
    if c[1] > 0.0
        for i1 = 1:n
            y[i1] = x[i1] * 2.0
        end
    else
        for i2 = 2:n
            u[i2] = 0.5 * u[i2 - 1] + x[i2]
        end
    end
    return nothing
end
