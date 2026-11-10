# f3_nest_III_fz
# Nest III, leading independent loops pre-fused.
# LOOP fz I K
function f3_nest_III_fz(x, y, a, n1, n2, n3)
    for fz = 1:n1 * n2 * n3
        l1 = mod(div(fz - 1, n2 * n3), n1) + 1
        l2 = mod(div(fz - 1, n3), n2) + 1
        l3 = mod(fz - 1, n3) + 1
        y[l1, l2, l3] = a * x[l1, l2, l3]
    end
    return nothing
end
