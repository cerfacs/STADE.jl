# f3_nest_SIIIS
# Nest SIIIS.
# LOOP l1 S H
# LOOP l2 I K
# LOOP l3 I T
# LOOP l4 I T
# LOOP l5 S T
function f3_nest_SIIIS(x, y, a, n1, n2, n3, n4, n5)
    for l1 = 2:n1
        for l2 = 1:n2
            for l3 = 1:n3
                for l4 = 1:n4
                    for l5 = 2:n5
                        y[l1, l2, l3, l4, l5] = 0.5 * y[l1 - 1, l2, l3, l4, l5] + 0.5 * y[l1, l2, l3, l4, l5 - 1] + x[l1, l2, l3, l4, l5]
                    end
                end
            end
        end
    end
    return nothing
end
