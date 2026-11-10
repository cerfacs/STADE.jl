# f3_nest_IIIS_fz
# Nest IIIS, leading independent loops pre-fused.
# LOOP fz I K
# LOOP l4 S T
function f3_nest_IIIS_fz(x, y, a, n1, n2, n3, n4)
    for fz = 1:n1 * n2 * n3
        l1 = mod(div(fz - 1, n2 * n3), n1) + 1
        l2 = mod(div(fz - 1, n3), n2) + 1
        l3 = mod(fz - 1, n3) + 1
        for l4 = 2:n4
            y[l1, l2, l3, l4] = 0.5 * y[l1, l2, l3, l4 - 1] + x[l1, l2, l3, l4]
        end
    end
    return nothing
end
