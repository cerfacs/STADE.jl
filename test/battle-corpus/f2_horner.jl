# f2_horner
# Horner evaluation.
# LOOP i1 S H
function f2_horner(c, x, out, n)
    s1 = 0.0
    for i1 = 1:n
        s1 = s1 * x[1] + c[i1]
    end
    out[1] = s1
    return nothing
end
