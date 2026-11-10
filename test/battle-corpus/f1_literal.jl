# f1_literal
# Literal bounds.
# LOOP i1 I K
function f1_literal(x, y)
    for i1 = 1:8
        y[i1] = x[i1] * x[i1]
    end
    return nothing
end
