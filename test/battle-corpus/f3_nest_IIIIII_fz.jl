# f3_nest_IIIIII_fz
# Nest IIIIII, leading independent loops pre-fused.
# LOOP fz I K
function f3_nest_IIIIII_fz(x, y, a, n1, n2, n3, n4, n5, n6)
    for fz = 1:n1 * n2 * n3 * n4 * n5 * n6
        l1 = mod(div(fz - 1, n2 * n3 * n4 * n5 * n6), n1) + 1
        l2 = mod(div(fz - 1, n3 * n4 * n5 * n6), n2) + 1
        l3 = mod(div(fz - 1, n4 * n5 * n6), n3) + 1
        l4 = mod(div(fz - 1, n5 * n6), n4) + 1
        l5 = mod(div(fz - 1, n6), n5) + 1
        l6 = mod(fz - 1, n6) + 1
        y[l1, l2, l3, l4, l5, l6] = a * x[l1, l2, l3, l4, l5, l6]
    end
    return nothing
end
