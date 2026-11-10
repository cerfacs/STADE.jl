# f3_nest_IISI_fz
# Nest IISI, leading independent loops pre-fused.
# LOOP fz I K
# LOOP l3 S T
# LOOP l4 I T
function f3_nest_IISI_fz(x, y, a, n1, n2, n3, n4)
    for fz = 1:n1 * n2
        l1 = mod(div(fz - 1, n2), n1) + 1
        l2 = mod(fz - 1, n2) + 1
        for l3 = 2:n3
            for l4 = 1:n4
                y[l1, l2, l3, l4] = 0.5 * y[l1, l2, l3 - 1, l4] + x[l1, l2, l3, l4]
            end
        end
    end
    return nothing
end
