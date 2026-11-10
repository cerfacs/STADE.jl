# f3_nest_SIS
# Nest SIS.
# LOOP l1 S H
# LOOP l2 I K
# LOOP l3 S T
function f3_nest_SIS(x, y, a, n1, n2, n3)
    for l1 = 2:n1
        for l2 = 1:n2
            for l3 = 2:n3
                y[l1, l2, l3] = 0.5 * y[l1 - 1, l2, l3] + 0.5 * y[l1, l2, l3 - 1] + x[l1, l2, l3]
            end
        end
    end
    return nothing
end
