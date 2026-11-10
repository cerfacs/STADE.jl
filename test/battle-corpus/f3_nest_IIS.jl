# f3_nest_IIS
# Nest IIS.
# LOOP l1 I K
# LOOP l2 I T
# LOOP l3 S T
function f3_nest_IIS(x, y, a, n1, n2, n3)
    for l1 = 1:n1
        for l2 = 1:n2
            for l3 = 2:n3
                y[l1, l2, l3] = 0.5 * y[l1, l2, l3 - 1] + x[l1, l2, l3]
            end
        end
    end
    return nothing
end
