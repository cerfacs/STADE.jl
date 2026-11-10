# f2_product
# Running product written out each step.
# LOOP i1 S H
function f2_product(x, y, n)
    p1 = 1.0
    for i1 = 1:n
        p1 = p1 * x[i1]
        y[i1] = p1
    end
    return nothing
end
