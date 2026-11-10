# f5_loop_in_branch
# Loops inside the two arms of a host branch.
# LOOP i1 I K
# LOOP i2 I K
function f5_loop_in_branch(c, x, y, z, n)
    if c[1] > 0.0
        for i1 = 1:n
            y[i1] = x[i1] * 2.0
        end
    else
        for i2 = 1:n
            z[i2] = x[i2] + 1.0
        end
    end
    return nothing
end
