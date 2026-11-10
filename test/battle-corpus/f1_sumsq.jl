# f1_sumsq
# Scalar reduction into a length-1 array.
# LOOP i1 I K
function f1_sumsq(x, out, n)
    s1 = 0.0
    for i1 = 1:n
        s1 = s1 + x[i1] * x[i1]
    end
    out[1] = s1
    return nothing
end
