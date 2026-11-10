# f2_scalar_carry
# Scalar carried between iterations, not a plain sum.
# LOOP i1 S H
function f2_scalar_carry(x, y, n)
    s1 = 0.0
    for i1 = 1:n
        s1 = 0.5 * s1 + x[i1]
        y[i1] = s1
    end
    return nothing
end
