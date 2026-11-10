# f3_nest_SSII
# Nest SSII.
# LOOP l1 S H
# LOOP l2 S H
# LOOP l3 I K
# LOOP l4 I T
function f3_nest_SSII(x, y, a, n1, n2, n3, n4)
    for l1 = 2:n1
        for l2 = 2:n2
            for l3 = 1:n3
                for l4 = 1:n4
                    y[l1, l2, l3, l4] = 0.5 * y[l1 - 1, l2, l3, l4] + 0.5 * y[l1, l2 - 1, l3, l4] + x[l1, l2, l3, l4]
                end
            end
        end
    end
    return nothing
end
