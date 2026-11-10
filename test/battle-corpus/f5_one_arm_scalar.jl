# f5_one_arm_scalar
# Scalar set in one arm only, private to the iteration.
# LOOP i1 I K
function f5_one_arm_scalar(x, y, n)
    for i1 = 1:n
        s1 = 0.0
        if x[i1] > 0.0
            s1 = x[i1]
        end
        y[i1] = s1
    end
    return nothing
end
