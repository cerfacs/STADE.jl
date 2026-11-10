# f3_nest_IIII_fz
# Nest IIII, leading independent loops pre-fused.
# LOOP fz I K
function f3_nest_IIII_fz(x, y, a, n1, n2, n3, n4)
    for fz = 1:n1 * n2 * n3 * n4
        l1 = mod(div(fz - 1, n2 * n3 * n4), n1) + 1
        l2 = mod(div(fz - 1, n3 * n4), n2) + 1
        l3 = mod(div(fz - 1, n4), n3) + 1
        l4 = mod(fz - 1, n4) + 1
        y[l1, l2, l3, l4] = a * x[l1, l2, l3, l4]
    end
    return nothing
end
