# f3_nest_ISISI
# Nest ISISI.
# LOOP l1 I K
# LOOP l2 S T
# LOOP l3 I T
# LOOP l4 S T
# LOOP l5 I T
function f3_nest_ISISI(x, y, a, n1, n2, n3, n4, n5)
    for l1 = 1:n1
        for l2 = 2:n2
            for l3 = 1:n3
                for l4 = 2:n4
                    for l5 = 1:n5
                        y[l1, l2, l3, l4, l5] = 0.5 * y[l1, l2 - 1, l3, l4, l5] + 0.5 * y[l1, l2, l3, l4 - 1, l5] + x[l1, l2, l3, l4, l5]
                    end
                end
            end
        end
    end
    return nothing
end
