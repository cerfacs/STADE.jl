# f3_nest_III
# Nest III.
# LOOP l1 I K
# LOOP l2 I T
# LOOP l3 I T
function f3_nest_III(x, y, a, n1, n2, n3)
    for l1 = 1:n1
        for l2 = 1:n2
            for l3 = 1:n3
                y[l1, l2, l3] = a * x[l1, l2, l3]
            end
        end
    end
    return nothing
end
