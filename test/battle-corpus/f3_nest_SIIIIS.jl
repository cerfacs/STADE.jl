# f3_nest_SIIIIS
# Nest SIIIIS.
# LOOP l1 S H
# LOOP l2 I K
# LOOP l3 I T
# LOOP l4 I T
# LOOP l5 I T
# LOOP l6 S T
function f3_nest_SIIIIS(x, y, a, n1, n2, n3, n4, n5, n6)
    for l1 = 2:n1
        for l2 = 1:n2
            for l3 = 1:n3
                for l4 = 1:n4
                    for l5 = 1:n5
                        for l6 = 2:n6
                            y[l1, l2, l3, l4, l5, l6] = 0.5 * y[l1 - 1, l2, l3, l4, l5, l6] + 0.5 * y[l1, l2, l3, l4, l5, l6 - 1] + x[l1, l2, l3, l4, l5, l6]
                        end
                    end
                end
            end
        end
    end
    return nothing
end
