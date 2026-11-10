# f3_nest_SSI_tri
# Nest SSI, triangular bounds.
# LOOP l1 S H
# LOOP l2 S H
# LOOP l3 I K
function f3_nest_SSI_tri(x, y, a, n1, n2, n3)
    for l1 = 2:n1
        for l2 = 2:l1
            for l3 = 1:l2
                y[l1, l2, l3] = 0.5 * y[l1 - 1, l2, l3] + 0.5 * y[l1, l2 - 1, l3] + x[l1, l2, l3]
            end
        end
    end
    return nothing
end
