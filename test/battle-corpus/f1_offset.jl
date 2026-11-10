# f1_offset
# Write at i1 + 1 (distinct, so independent).
# LOOP i1 I K
function f1_offset(x, y, n)
    for i1 = 1:n
        y[i1 + 1] = x[i1] * 2.0
    end
    return nothing
end
