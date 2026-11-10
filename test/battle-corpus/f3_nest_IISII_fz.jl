# f3_nest_IISII_fz
# Nest IISII, leading independent loops pre-fused.
# LOOP fz I K
# LOOP l3 S T
# LOOP l4 I T
# LOOP l5 I T
function f3_nest_IISII_fz(x, y, a, n1, n2, n3, n4, n5)
    for fz = 1:n1 * n2
        l1 = mod(div(fz - 1, n2), n1) + 1
        l2 = mod(fz - 1, n2) + 1
        for l3 = 2:n3
            for l4 = 1:n4
                for l5 = 1:n5
                    y[l1, l2, l3, l4, l5] = 0.5 * y[l1, l2, l3 - 1, l4, l5] + x[l1, l2, l3, l4, l5]
                end
            end
        end
    end
    return nothing
end
