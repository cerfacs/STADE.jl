# f2_prefix
# Prefix sum.
# LOOP i1 S H
function f2_prefix(x, c, n)
    c[1] = x[1]
    for i1 = 2:n
        c[i1] = c[i1 - 1] + x[i1]
    end
    return nothing
end
