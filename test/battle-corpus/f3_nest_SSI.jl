# f3_nest_SSI
# Nest SSI.
# LOOP l1 S H
# LOOP l2 S H
# LOOP l3 I K
function f3_nest_SSI(x, y, a, n1, n2, n3)
    for l1 = 2:n1
        for l2 = 2:n2
            for l3 = 1:n3
                y[l1, l2, l3] = 0.5 * y[l1 - 1, l2, l3] + 0.5 * y[l1, l2 - 1, l3] + x[l1, l2, l3]
            end
        end
    end
    return nothing
end
