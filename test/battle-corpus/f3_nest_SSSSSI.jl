# f3_nest_SSSSSI
# Nest SSSSSI.
# LOOP l1 S H
# LOOP l2 S H
# LOOP l3 S H
# LOOP l4 S H
# LOOP l5 S H
# LOOP l6 I K
function f3_nest_SSSSSI(x, y, a, n1, n2, n3, n4, n5, n6)
    for l1 = 2:n1
        for l2 = 2:n2
            for l3 = 2:n3
                for l4 = 2:n4
                    for l5 = 2:n5
                        for l6 = 1:n6
                            y[l1, l2, l3, l4, l5, l6] = 0.5 * y[l1 - 1, l2, l3, l4, l5, l6] + 0.5 * y[l1, l2 - 1, l3, l4, l5, l6] + 0.5 * y[l1, l2, l3 - 1, l4, l5, l6] + 0.5 * y[l1, l2, l3, l4 - 1, l5, l6] + 0.5 * y[l1, l2, l3, l4, l5 - 1, l6] + x[l1, l2, l3, l4, l5, l6]
                        end
                    end
                end
            end
        end
    end
    return nothing
end
