# f3_nest_ISSI
# Nest ISSI.
# LOOP l1 I K
# LOOP l2 S T
# LOOP l3 S T
# LOOP l4 I T
function f3_nest_ISSI(x, y, a, n1, n2, n3, n4)
    for l1 = 1:n1
        for l2 = 2:n2
            for l3 = 2:n3
                for l4 = 1:n4
                    y[l1, l2, l3, l4] = 0.5 * y[l1, l2 - 1, l3, l4] + 0.5 * y[l1, l2, l3 - 1, l4] + x[l1, l2, l3, l4]
                end
            end
        end
    end
    return nothing
end
