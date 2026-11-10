# f3_nest_IIII
# Nest IIII.
# LOOP l1 I K
# LOOP l2 I T
# LOOP l3 I T
# LOOP l4 I T
function f3_nest_IIII(x, y, a, n1, n2, n3, n4)
    for l1 = 1:n1
        for l2 = 1:n2
            for l3 = 1:n3
                for l4 = 1:n4
                    y[l1, l2, l3, l4] = a * x[l1, l2, l3, l4]
                end
            end
        end
    end
    return nothing
end
