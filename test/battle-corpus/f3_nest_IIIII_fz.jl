# f3_nest_IIIII_fz
# Nest IIIII, leading independent loops pre-fused.
# LOOP fz I K
function f3_nest_IIIII_fz(x, y, a, n1, n2, n3, n4, n5)
    for fz = 1:n1 * n2 * n3 * n4 * n5
        l1 = mod(div(fz - 1, n2 * n3 * n4 * n5), n1) + 1
        l2 = mod(div(fz - 1, n3 * n4 * n5), n2) + 1
        l3 = mod(div(fz - 1, n4 * n5), n3) + 1
        l4 = mod(div(fz - 1, n5), n4) + 1
        l5 = mod(fz - 1, n5) + 1
        y[l1, l2, l3, l4, l5] = a * x[l1, l2, l3, l4, l5]
    end
    return nothing
end
