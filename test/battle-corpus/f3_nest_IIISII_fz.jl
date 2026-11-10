# f3_nest_IIISII_fz
# Nest IIISII, leading independent loops pre-fused.
# LOOP fz I K
# LOOP l4 S T
# LOOP l5 I T
# LOOP l6 I T
function f3_nest_IIISII_fz(x, y, a, n1, n2, n3, n4, n5, n6)
    for fz = 1:n1 * n2 * n3
        l1 = mod(div(fz - 1, n2 * n3), n1) + 1
        l2 = mod(div(fz - 1, n3), n2) + 1
        l3 = mod(fz - 1, n3) + 1
        for l4 = 2:n4
            for l5 = 1:n5
                for l6 = 1:n6
                    y[l1, l2, l3, l4, l5, l6] = 0.5 * y[l1, l2, l3, l4 - 1, l5, l6] + x[l1, l2, l3, l4, l5, l6]
                end
            end
        end
    end
    return nothing
end
