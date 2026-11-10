# f3_nest_ISI_tri
# Nest ISI, triangular bounds.
# LOOP l1 I K
# LOOP l2 S T
# LOOP l3 I T
function f3_nest_ISI_tri(x, y, a, n1, n2, n3)
    for l1 = 1:n1
        for l2 = 2:l1
            for l3 = 1:l2
                y[l1, l2, l3] = 0.5 * y[l1, l2 - 1, l3] + x[l1, l2, l3]
            end
        end
    end
    return nothing
end
