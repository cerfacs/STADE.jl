# f3_nest_SISI
# Nest SISI.
# LOOP l1 S H
# LOOP l2 I K
# LOOP l3 S T
# LOOP l4 I T
function f3_nest_SISI(x, y, a, n1, n2, n3, n4)
    for l1 = 2:n1
        for l2 = 1:n2
            for l3 = 2:n3
                for l4 = 1:n4
                    y[l1, l2, l3, l4] = 0.5 * y[l1 - 1, l2, l3, l4] + 0.5 * y[l1, l2, l3 - 1, l4] + x[l1, l2, l3, l4]
                end
            end
        end
    end
    return nothing
end
