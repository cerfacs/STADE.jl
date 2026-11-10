# f1_scalar_priv
# Scalar written then read inside the body: private, not carried.
# LOOP i1 I K
function f1_scalar_priv(x, y, n)
    for i1 = 1:n
        t1 = x[i1] * x[i1]
        t1 = t1 + 1.0
        y[i1] = sqrt(t1)
    end
    return nothing
end
