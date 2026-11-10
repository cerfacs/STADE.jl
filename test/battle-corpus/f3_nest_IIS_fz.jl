# f3_nest_IIS_fz
# Nest IIS, leading independent loops pre-fused.
# LOOP fz I K
# LOOP l3 S T
function f3_nest_IIS_fz(x, y, a, n1, n2, n3)
    for fz = 1:n1 * n2
        l1 = mod(div(fz - 1, n2), n1) + 1
        l2 = mod(fz - 1, n2) + 1
        for l3 = 2:n3
            y[l1, l2, l3] = 0.5 * y[l1, l2, l3 - 1] + x[l1, l2, l3]
        end
    end
    return nothing
end
