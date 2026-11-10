# f3_nest_ISSSS
# Nest ISSSS.
# LOOP l1 I K
# LOOP l2 S T
# LOOP l3 S T
# LOOP l4 S T
# LOOP l5 S T
function f3_nest_ISSSS(x, y, a, n1, n2, n3, n4, n5)
    for l1 = 1:n1
        for l2 = 2:n2
            for l3 = 2:n3
                for l4 = 2:n4
                    for l5 = 2:n5
                        y[l1, l2, l3, l4, l5] = 0.5 * y[l1, l2 - 1, l3, l4, l5] + 0.5 * y[l1, l2, l3 - 1, l4, l5] + 0.5 * y[l1, l2, l3, l4 - 1, l5] + 0.5 * y[l1, l2, l3, l4, l5 - 1] + x[l1, l2, l3, l4, l5]
                    end
                end
            end
        end
    end
    return nothing
end
